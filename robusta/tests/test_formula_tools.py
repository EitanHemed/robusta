import pytest

import robusta as rst


# Variables Parser
def test_many_spaces():
    vp = rst.misc.formula_tools.VariablesParser(
        'score~        group   * cond   + time | id')
    assert vp.get_variables() == ('score', ['group', 'cond'], ['time'], 'id')


def test_no_between_subject_term():
    vp = rst.misc.formula_tools.VariablesParser('score ~ time|id')
    assert vp.get_variables() == ('score', [], ['time'], 'id')


def test_no_within_subject_term():
    vp = rst.misc.formula_tools.VariablesParser('score ~ group + 1|id')
    assert vp.get_variables() == ('score', ['group'], [], 'id')


def test_missing_terms():
    # Todo - rewrite all testing of incorrect specification of formula
    with pytest.assertRaises(RuntimeError):
        # No subject term
        rst.misc.formula_tools.VariablesParser('score ~ group'
                                               ).get_variables()
    with pytest.assertRaises(RuntimeError):
        # No independent variables
        rst.misc.formula_tools.VariablesParser('score ~ group'
                                               ).get_variables()
    with pytest.assertRaises(RuntimeError):
        # No dependent variable
        rst.misc.formula_tools.VariablesParser('score').get_variables()
    with pytest.assertRaises(RuntimeError):
        rst.misc.formula_tools.VariablesParser(
            'score + time|id').get_variables()


# Formula Parser
def test_all_terms():
    fp = rst.misc.formula_tools.FormulaParser('luck', ['pj'], ['volume'], 's')
    assert 'luck~pj+(volume|s)' == fp.get_formula()


def test_no_beteen_subject_term():
    fp = rst.misc.formula_tools.FormulaParser('luck', [], ['volume'], 's')
    assert 'luck~(volume|s)' == fp.get_formula()


def test_many_terms():
    fp = rst.misc.formula_tools.FormulaParser(
        'luck', ['h', 'pj'], ['volume', 'page'], 's')
    assert 'luck~h*pj+(volume*page|s)' == fp.get_formula()


def test_missing_terms():
    fp = rst.misc.formula_tools.FormulaParser(
        'luck', ['h', 'pj'], ['volume', 'page'], 's')
    assert 'luck~h*pj+(volume*page|s)' == fp.get_formula()
