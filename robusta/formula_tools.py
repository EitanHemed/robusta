import re

from patsy.desc import ModelDesc


class VariablesParser():

    def __init__(self, formula):

        self.patsy_parsed = ModelDesc.from_formula(formula)
        self.dependent, self.between, self.within, self.subject = (
            self.parse_variables_from_formula())

    def parse_variables_from_formula(self):
        # This one is the easiest
        between = []
        within = []
        subject = None

        for t in self.patsy_parsed.rhs_termlist:
            # We expect something along the lines of (1|subject) or
            # even (within_variable1|within_variable_2|id_variable)
            # so a pipe symbol gives us the id and within subject factor(s)
            # clause.
            if '|' in t.name():
                pipe_count = t.name().count('|')
                # Number of within subject factors is <= 1
                if pipe_count == 1:
                    # Last one should be subject
                    _ws, subject = t.name().split('|')
                    # Drop leading whitespace
                    subject = subject.strip()
                    # No within subject factor
                    if _ws.strip() == '1': #bool(re.match('\s*1\s*', _)):
                        pass
                    # One within subject factor
                    else:
                        within.append(_ws.strip())
                # More than one within-subject factor:
                else:
                    within.extend([i.strip() for i in _ws.split('|')])
            # A between subject term
            else:
                # An interaction, skip it
                if len(t.factors) > 1 and ':' in t.name():
                    pass
                # A valid between subject term
                else:
                    between.append(t.name())


        # Patsy automatically adds this one, which we don't need
        between.remove('Intercept')

        # Now we can test the parsed variables
        try:
            dependent = self.patsy_parsed.lhs_termlist[0].name()
        except IndexError:
            raise RuntimeError('No dependent variable defined. Use a formula'
                               'similar to y~x')
        if within + between == []:  #
            raise RuntimeError('No independent variables defined')
        if subject is None:
            raise RuntimeError('No subject term defined')

        return dependent, between, within, subject

    def get_variables(self):
        return self.dependent, self.between, self.within, self.subject


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
