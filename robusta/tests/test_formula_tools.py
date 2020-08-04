import unittest
import robusta as rst


class TestVariableParser(unittest.TestCase):

    def test_many_spaces(self):
        vp = rst.formula_tools.VariablesParser(
            'score~        group   * cond   + time | id')
        self.assertEqual(vp.get_variables(),
                         ('score', ['group', 'cond'], ['time'], 'id'))

    def test_no_between_subject_term(self):
        vp = rst.formula_tools.VariablesParser('score ~ time|id')
        self.assertEqual(vp.get_variables(), ('score', [], ['time'], 'id'))

    def test_no_within_subject_term(self):
        vp = rst.formula_tools.VariablesParser('score ~ group + 1|id')
        self.assertEqual(vp.get_variables(), ('score', ['group'], [], 'id'))

    def test_missing_terms(self):
        # Todo - rewrite all testing of incorrect specification of formula
        return
        with self.assertRaises(RuntimeError):
            # No subject term
            rst.formula_tools.VariablesParser('score ~ group'
                                              ).get_variables()
            # No independent variables
            rst.formula_tools.VariablesParser('score ~ group'
                                              ).get_variables()
            # No dependent variable
            rst.formula_tools.VariablesParser('score').get_variables()
            rst.formula_tools.VariablesParser('score + time|id').get_variables()


class TestFormulaParser(unittest.TestCase):

    def test_all_terms(self):
        fp = rst.formula_tools.FormulaParser('luck', ['pj'], ['volume'], 's')
        self.assertEqual('luck~pj+(volume|s)', fp.get_formula())

    def test_no_beteen_subject_term(self):
        fp = rst.formula_tools.FormulaParser('luck', [], ['volume'], 's')
        self.assertEqual('luck~(volume|s)', fp.get_formula())

    def test_many_terms(self):
        fp = rst.formula_tools.FormulaParser(
            'luck', ['h', 'pj'], ['volume', 'page'], 's')
        self.assertEqual('luck~h*pj+(volume*page|s)', fp.get_formula())

    def test_missing_terms(self):
        fp = rst.formula_tools.FormulaParser(
            'luck', ['h', 'pj'], ['volume', 'page'], 's')
        self.assertEqual('luck~h*pj+(volume*page|s)', fp.get_formula())



if __name__ == '__main__':
    unittest.main()
