import re
import string
import warnings

import pyparsing
from patsy.desc import ModelDesc


class VariablesParser:

    def __init__(self, formula):
        self.formula = formula
        self.parsed_formula = self.parse_regression_formula()
        self.assign_variables_from_formula()
        self.test_and_finalize_variables()

    def assign_variables_from_formula(self):
        # Todo - rewrite all incorrect specification of formula and
        _d = self.parsed_formula.asDict()
        self.dependent = _d['dependent'][0]
        self.between = _d.get('between', [])
        self.within = _d.get('within')
        self.subject = _d['subject']

    def test_and_finalize_variables(self):
        if self.subject == '':
            raise RuntimeError('No subject variables defined')
        if (self.within + self.between) == []:
            raise RuntimeError('No between or within variables defined')

        # TODO - This is an ugly patch that designs with no within-subject term
        #  the handling should occur within the parser definition
        if self.within == ['1']:
            self.within = []

        # return
        #
        # # This one is the easiest
        # between = []
        # within = []
        # subject = None
        #
        # for t in self.patsy_parsed.rhs_termlist:
        #     # We expect something along the lines of (1|subject) or
        #     # even (within_variable1|within_variable_2|id_variable)
        #     # so a pipe symbol gives us the id and within subject factor(s)
        #     # clause.
        #     if '|' in t.name():
        #         pipe_count = t.name().count('|')
        #         # Number of within subject factors is <= 1
        #         if pipe_count == 1:
        #             # Last one should be subject
        #             _ws, subject = t.name().split('|')
        #             # Drop leading whitespace
        #             subject = subject.strip()
        #             # No within subject factor
        #             if _ws.strip() == '1':  # bool(re.match('\s*1\s*', _)):
        #                 pass
        #             # One within subject factor
        #             else:
        #                 within.append(_ws.strip())
        #         # More than one within-subject factor:
        #         else:
        #             within.extend([i.strip() for i in _ws.split('|')])
        #     # A between subject term
        #     else:
        #         # An interaction, skip it
        #         if len(t.factors) > 1 and ':' in t.name():
        #             pass
        #         # A valid between subject term
        #         else:
        #             between.append(t.name())
        #
        # # Patsy automatically adds this one, which we don't need
        # between.remove('Intercept')
        #
        # # Now we can test the parsed variables
        # try:
        #     dependent = self.patsy_parsed.lhs_termlist[0].name()
        # except IndexError:
        #     raise RuntimeError('No dependent variable defined. Use a formula'
        #                        'similar to y~x')
        # if within + between == []:  #
        #     raise RuntimeError('No independent variables defined')
        # if subject is None:
        #     raise RuntimeError('No subject term defined')
        #
        # return dependent, between, within, subject

    def get_variables(self):
        return self.dependent, self.between, self.within, self.subject

    # def get_formula(self):
    #    return self.patsy_parsed

    def parse_regression_formula(self):

        # A possible variable name is:
        # Alphanumeirc + sum non-meaningful punctuation marks
        varname = pyparsing.Word(
            pyparsing.alphanums + re.sub('[+*$|()~]', '', string.punctuation))
        # We need to identify the dependent, between, within and subject variables
        # First we define the patterns
        # Dependent variable is the stub in the form of 'y~'
        dependent = (varname + pyparsing.Suppress('~')).setResultsName(
            'dependent')

        # Usually the dependent is followed by the between-subject terms.
        # Between terms are optional (looking something like dose + condition)
        between_terms = pyparsing.ZeroOrMore(
                varname + pyparsing.Suppress(pyparsing.oneOf(['+', '*']))
        ).setResultsName('between')

        # There are two possible patterns for within and subject variables
        # 1. Only subject variable (e.g., 1|ID), no within-term
        # 2. Within and subject (such as condition|order|ID)
        within_and_subject_terms = (
                (pyparsing.ZeroOrMore(pyparsing.Suppress('(')))
                + (pyparsing.OneOrMore(
                    pyparsing.OneOrMore(
                        varname + pyparsing.Suppress('|'))
                    | pyparsing.Suppress('1'))).setResultsName('within')
                + varname.setResultsName('subject')
                + (pyparsing.ZeroOrMore(pyparsing.Suppress(')')))
        )

        prsr = (dependent + between_terms + within_and_subject_terms )
        return prsr.parseString(self.formula)


class FormulaParser:

    def __init__(self, dependent, between, within, subject):
        self.formula = self.parse_formula_from_variables(
            dependent, between, within, subject
        )

    def parse_formula_from_variables(self, dependent, between, within, subject):
        if within == []:
            _ws = f'(1|{subject})'
        else:
            _ws = f"({'*'.join(within)}|{subject})"

        if between == []:
            _bs = ''
        else:
            _bs = f"{'*'.join(between)}+"

        return f'{dependent}~{_bs}{_ws}'

    def get_formula(self):
        return self.formula

