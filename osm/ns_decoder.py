"""
Resources for the decoding of *ns* encoded "numbersets" (n0, n1, n2, n3).

*ns* is a encoding scheme for tuples (n0, n1, n2, n3) with non-negative numbers n0, …
For instance, the encoding of (974, 2023, 898, 96) is "974 2 023 (67%) 898 (30%) 96 (3%)".
It is used in the user statistics provided by
`How did you contribute to OpenStreetMap? <https://hdyc.neis-one.org/>`_ (when column delimiters are ignored):
Here n1, n2 and n3 are the numbers of the users's create-, modify- and delete-edits of OSM objects
(nodes, ways, relations); n0 is the corresponding number of OSM objects which were last modified by the user.

An *ns* string can be derived from the goal symbol *numberset* by the following grammar.

* Character classes

  * d1 := "1" | … | "9"
  * d := "0" | d1

* Tokens

  * BLANK := " "
  * ZERO  := "0"
  * NA    := d1 [d]
  * NB    := "0" d d
  * NAB   := d1 d d
  * LPAR  := "("
  * RPAR  := "%)"

* Non-terminals

  * number := (ZERO BLANK) | ( (NA | NAB) BLANK {(NB | NAB) BLANK} )
  * percent := LPAR (ZERO | NA | NAB) RPAR
  * num_percent := number percent
  * numberset := number num_percent BLANK num_percent BLANK num_percent

The numbers and percentages in the string satisfy the conditions

* n0 <= n1 + n2 + n3
  (not verified if semantic check is "loose")
* The percentages in the string are 100 \* n1/(n1+n2+n3), …

.. note:: In most cases, the decoding is unambiguous. A counterexample is "1 222 333 444 (100%) 0 (0%) 0 (0%)",
   which may be decoded as (1, 222333444, 0, 0) or as (1222, 333444, 0, 0).

:class Token: token with additional info such as value (if token is numeric) or column number.
:class Scanner: provides methods to decompose a string into tokens.
:class Parser: decodes an *ns* string.
:function decode_csv: decodes *ns* strings from a csv file.

.. todo::

   * emit warning/error if string allows more than one decoding
"""

import sys
import csv
from enum import Enum
from typing import List, Optional, TextIO, Tuple


class Token:
    """
    Token of an *ns* string.

    :attribute type: the type of this token.
    :type type: Token.Type
    :attribute numeric: "the token has a numeric value" (= self.type in (ZERO, NA, NB, or NAB)).
    :type numeric: bool
    :attribute val: the numeric value (if self.numeric).
    :type val: int
    :attribute col: the column number in the source string (or -1 if unknown)
    :type col: int
    """

    class Type(Enum):
        """
        Token types from the grammar (BLANK, ZERO, NA, NB, NAB, LPAR, RPAR) as well as EOS (End Of Source) and INVALID.
        """
        BLANK, ZERO, NA, NB, NAB, LPAR, RPAR, EOS, INVALID = range(1, 10)

    def __init__(self,
                 ttype: Type,
                 val: int = -1,
                 col: int = -1) -> None:
        """
        Initialize a new instance.

        :param ttype: token type
        :param val: numeric value (for ZERO, NA, NB and NAB tokens) or -1
        :param col: column number of the token or -1
        """
        self.type: Token.Type
        self.numeric: bool
        self.val: int
        self.col: int
        val_ok: bool
        # assert 0 <= ttype <= 8, "unknown token type " + str(ttype)
        if ttype == Token.Type.ZERO:
            val_ok = val == -1 or val == 0  # val is optional
        elif ttype == Token.Type.NA:
            val_ok = 1 <= val <= 99
        elif ttype == Token.Type.NB:
            val_ok = 0 <= val <= 99
        elif ttype == Token.Type.NAB:
            val_ok = 100 <= val <= 999
        else:
            val_ok = val == -1
        assert val_ok,  "invalid value " + str(val) + " for token " + ttype.name
        assert col >= -1, "invalid column number " + str(col)
        self.type = ttype
        self.col = col
        self.numeric = (ttype in (Token.Type.ZERO, Token.Type.NA, Token.Type.NB, Token.Type.NAB))
        if self.numeric:
            self.val = max(0, val)  # val may be -1 for Token.Type.ZERO

    def __str__(self) -> str:
        return self.type.name\
               + ("[" + str(self.val) + "]" if self.numeric else "")\
               + ("@c" + str(self.col) if self.col != -1 else "")


class Scanner:
    """
    Lexical analysis of a source text (string).

    :attribute _line: the source text.
    :type _line: str
    :attribute _col: column number of first character after current lookahead
    :type _col: int
    :attribute loah: the current lookahead.
    :type loah: Token

    >>> sc = Scanner("0 123 (45%)")
    >>> print(sc.loah, sc.next_lookahead(), sc.next_lookahead(), sc.loah)
    ZERO[0]@c0 BLANK@c1 NAB[123]@c2 NAB[123]@c2
    >>> sc.print_all_tokens()
    ZERO[0]@c0 BLANK@c1 NAB[123]@c2 BLANK@c5 LPAR@c6 NA[45]@c7 RPAR@c9 EOS@c11
    """

    _EOS_CHAR: str = "$"

    def __init__(self,
                 line: str) -> None:
        """
        Initialize a new instance.

        :param line: the source text.
        """
        self._line: str = line
        self._col: int = 0  # col number of first char of next lookahead
        self.loah: Token = self.next_lookahead()  # current lookahead

    def reset(self) -> None:
        """
        Resets the scanner to the initial state (self.loah = first token).
        """
        self._col = 0
        self.loah = self.next_lookahead()

    def _cur_char(self) -> str:
        assert 0 <= self._col <= len(self._line), "invalid column number " + str(self._col) \
                                                  + "; attempt to read after EOS (End of Source)?"
        return self._line[self._col] if self._col < len(self._line) else Scanner._EOS_CHAR


    def next_lookahead(self) -> Token:
        """
        Determine the next token in the source and return it.

        :return: the new current lookahead.
        """
        tcol: int = self._col
        ch: str = self._cur_char()
        new_loah: Token = Token(Token.Type.INVALID, col=tcol)
        if ch == " ":
            new_loah = Token(Token.Type.BLANK, col=tcol)
        elif "0" <= ch <= "9":
            num_len: int
            val: int
            while "0" <= self._cur_char() <= "9":
                self._col += 1
            self._col -= 1  # col number of last char of new token
            num_len = self._col - tcol + 1
            val = int(self._line[tcol:tcol + num_len])
            if num_len == 1 and val == 0:
                new_loah = Token(Token.Type.ZERO, col=tcol)
            elif num_len <= 2:
                new_loah = Token(Token.Type.NA, val, col=tcol)
            elif num_len == 3 and val < 100:
                new_loah = Token(Token.Type.NB, val, col=tcol)
            elif num_len == 3:
                new_loah = Token(Token.Type.NAB, val, col=tcol)
            del num_len, val
        elif ch == "(":
            new_loah = Token(Token.Type.LPAR, col=tcol)
        elif ch == "%":
            self._col += 1
            if self._cur_char() == ")":
                new_loah = Token(Token.Type.RPAR, col=tcol)
        elif ch == Scanner._EOS_CHAR:
            new_loah = Token(Token.Type.EOS, col=tcol)
        self.loah = new_loah
        self._col += 1  # col number of first char after loah
        return new_loah

    def print_all_tokens(self) -> None:
        """
        Prints all tokens in the source text (ending with Token.Type.EOS) and resets the scanner.
        """
        self.reset()
        while self.loah.type != Token.Type.EOS:
            print(str(self.loah), end=" ")
            self.next_lookahead()
        print(str(self.loah))
        self.reset()


class Parser:
    """
    Syntactical and semantical analysis of an *ns* string.

    :attribute scanner: the lexical analyser of an *ns* string.
    :type scanner: Scanner

    >>> ns = Parser("19 267 20 276 (86%) 2 650 (11%) 539 (2%)").parse_string()
    >>> print(ns)
    ((19267, 20276, 2650, 539), (-1, 86, 11, 2))
    """

    NP = Tuple[int, int]  # number, percentage
    N4 = Tuple[int, int, int, int]
    """Four numbers (n0, n1, n2, n3) or four percentages (p0=-1, p1, p2, p3)."""

    NS = Tuple[N4, N4]
    """Abstract representation of an *ns* string: ((n0, n1, n2, n3), (p0=-1, p1, p2, p3))."""

    def __init__(self,
                 ns_string: str) -> None:
        """
        Initialize a new instance.

        :param ns_string: the source text.
        """
        self.scanner: Scanner = Scanner(ns_string)

    def blank(self) -> None:
        if self.scanner.loah.type == Token.Type.BLANK:
            self.scanner.next_lookahead()
        else:
            raise SyntaxError("in blank(): BLANK expected, but " + str(self.scanner.loah) + " found.")

    def number(self,
               max_fblocks: int = sys.maxsize) -> int:
        val: int
        if self.scanner.loah.type == Token.Type.ZERO:
            val = self.scanner.loah.val
            self.scanner.next_lookahead()
            self.blank()
        elif self.scanner.loah.type in (Token.Type.NA, Token.Type.NAB):
            val = self.scanner.loah.val
            self.scanner.next_lookahead()
            self.blank()
            while max_fblocks > 0 and self.scanner.loah.type in (Token.Type.NB, Token.Type.NAB):
                max_fblocks -= 1
                val = val*1000 + self.scanner.loah.val
                self.scanner.next_lookahead()
                self.blank()
        else:
            raise SyntaxError("in number(): ZERO, NA or NAB as first token expected, but "
                              + str(self.scanner.loah) + " found.")
        return val

    def percent(self) -> int:
        pval: int
        if self.scanner.loah.type == Token.Type.LPAR:
            self.scanner.next_lookahead()
        else:
            raise SyntaxError("in percent(): LPAR expected, but " + str(self.scanner.loah) + " found.")
        if self.scanner.loah.type in (Token.Type.ZERO, Token.Type.NA, Token.Type.NAB):
            pval = self.scanner.loah.val
            if pval > 100:
                raise ValueError("in percent(): value = " + str(pval) + " > 100.")
            self.scanner.next_lookahead()
        else:
            raise SyntaxError("in percent(): ZERO, NA or NAB expected, but " + str(self.scanner.loah) + " found.")
        if self.scanner.loah.type == Token.Type.RPAR:
            self.scanner.next_lookahead()
        else:
            raise SyntaxError("in percent(): RPAR expected, but " + str(self.scanner.loah) + " found.")
        return pval

    def num_percent(self) -> NP:
        val: int = self.number()
        percent: int = self.percent()
        return (val, percent)

    def check_semantic(self,
                       numberset: NS,
                       strict: bool) -> bool:
        nums: Parser.N4 = numberset[0]
        percents: Parser.N4 = numberset[1]
        n_edits: int = nums[1] + nums[2] + nums[3]
        result: bool = not strict or (nums[0] <= n_edits)
        if n_edits > 0:
            for i in range(1, 4):
                result = result and abs(100*(nums[i]/n_edits) - percents[i]) < 0.501
        return result

    def numberset(self,
                  strict: bool) -> NS:
        valid_syntax_found: bool = False
        valid_semantic_found: bool = False
        terminate: bool = False
        max_n0_fblocks: int = sys.maxsize  # max number of digit blocks after first block in n0 (first number)
        result: Parser.NS
        while not valid_semantic_found and not terminate and max_n0_fblocks >= 0:
            n0: int = self.number(max_fblocks=max_n0_fblocks)
            if self.scanner.loah.type not in (Token.Type.NB, Token.Type.LPAR):
                terminate = (self.scanner.loah.type in (Token.Type.ZERO, Token.Type.NA))
                    # second <number> must start here; first <number> cannot be shorter.
                np1: Parser.NP = self.num_percent()
                self.blank()
                np2: Parser.NP = self.num_percent()
                self.blank()
                np3: Parser.NP = self.num_percent()
                valid_syntax_found = True
                result = ((n0, np1[0], np2[0], np3[0]), (-1, np1[1], np2[1], np3[1]))
                valid_semantic_found = self.check_semantic(result, strict)
                del np1, np2, np3
            if not valid_semantic_found and max_n0_fblocks >= 0:
                max_n0_fblocks = -1
                while n0 >= 1000:
                    max_n0_fblocks += 1
                    n0 //= 1000
                self.scanner.reset()
            del n0
        if not valid_semantic_found:
            if valid_syntax_found:
                msg: str
                if strict:
                    msg = "strict semantic check (n0 <= n1+n2+n3, matching percentages) fails."
                else:
                    msg = "loose semantic check (matching percentages) fails."
                raise ValueError("in numberset(): " + msg)
            else:
                raise SyntaxError("in numberset(): only one <number> before first LPAR.")
        return result

    def parse_string(self,
                     strict: bool = True) -> NS:
        """
        Parses the string with which this instance was initialized.

        :param strict: the condition n0 <= n1+n2+n3 will be enforced
        :return: an abstract representation of the string.
        :rtype: NS
        :raise SyntaxError: if the string isn't an *ns* string.
        :raise ValueError: if the constraints aren't satisfied.
        """
        result: Parser.NS = self.numberset(strict)
        if self.scanner.loah.type != Token.Type.EOS:
            raise SyntaxError("in parse_string(): EOS expected"
                              ", but " + str(self.scanner.loah) + " found.")
        assert result[1][0] == -1, "invalid percents[0] in result: " + str(result) + "[1][0] != -1."
        assert self.check_semantic(result, strict), "result " + str(result) + " violates semantic constraints."
        return result


def decode_csv(
        in_csv: Optional[str] = None,
        out_csv: Optional[str] = None) -> None:
    """
    Decodes *ns* strings from a csv file.

    The function parses the last entry of every row in a tab-separated input csv file.
    If it succeeds (the entry is an *ns* string), it appends the four numbers n0, …, n3 to the row.
    If it fails, it will instead append an error message.
    The resulting rows are output in csv format.

    :param in_csv: path name of csv file with *ns* strings in the last column (if None, the user will be prompted).
    :param out_csv: path name of output file (if None, the user will be prompted).
    :return: None
    """
    if in_csv is None:
        in_csv = str(input("Enter path of csv input file: "))
    if out_csv is None:
        out_csv = str(input("Enter path of csv output file (will be overwritten): "))
    in_file: TextIO
    out_file: TextIO
    with open(in_csv, "r", newline="") as in_file,\
         open(out_csv, "w", newline="") as out_file:
        reader = csv.reader(in_file, delimiter="\t")  # type is _csv._reader (cf. reveal_type() of mypy)
        writer = csv.writer(out_file, delimiter="\t")  # type is _csv._writer (cf. reveal_type() of mypy)
        row: List[str]
        for row in reader:
            ns: Optional[Parser.NS] = None
            err_message: str = ""
            try:
                try:
                    ns = Parser(row[-1]).parse_string(strict=True)  # <numberset> in last column
                except ValueError as exc:
                    err_message = str(type(exc).__name__) + " " + str(exc)
                    # if parsing fails with strict semantic check, try it with loose checks
                    ns = Parser(row[-1]).parse_string(strict=False)
            except (SyntaxError, ValueError) as exc:
                err_message = str(type(exc).__name__) + " " + str(exc)
            if ns is not None:
                i: int
                for i in range(0, 4):
                    row.append(str(ns[0][i]))  # append numbers only
                del i
            else:
                row += ["", "", "", ""]
            if err_message != "":
                row.append(err_message)
            writer.writerow(row)
            del ns
        del row
    del reader, writer, in_file, out_file


if __name__ == '__main__':

    choice = 3

    if choice == 1:
        Scanner("270 600 (82% %) 57 (1%) 5 (2%)").print_all_tokens()
        Scanner("270 600 297 43 (82%) 57 933 (1%) 5 995 (2%)").print_all_tokens()
    elif choice == 2:
        ns = Parser("270 636 297 243 (82%) 57 933 (16%) 5 995 (2%)").parse_string()
        print(str(ns))
    elif choice == 3:
        if len(sys.argv) != 3:
            print(f"Usage: python3 {sys.argv[0]} <in_csv> <out_csv>")
            sys.exit(-1)
        decode_csv(sys.argv[1], sys.argv[2])
