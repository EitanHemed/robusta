import numpy as np

from . import groupwise

P_VALUE_CLAUSE = 'p {pvalue_operator} {p.value:.3f}'
FREQ_T_CLAUSE = "{method} {alternative} t({parameter:.0f}) = {statistics:.2f}, " + P_VALUE_CLAUSE
COHEN_D_CLAUSE = "Cohen's d = {cohen_d:.2f}"
FREQ_T_DIFFERENCE_CLAUSE = "Mean Difference = {mean_diff:.2f}"

F_CLAUSE = "{term} F({df1:.0f}, {df2:.0f}) = {f:.2f}, " + P_VALUE_CLAUSE

BAYES_T_CLAUSE_DEC_NOTATION = "Bayes factor = {bf:.2f}, Error {error_operator} {error:3.f}%"
BAYES_T_CLAUSE_SCI_NOTATION = "Bayes factor = {bf:.2E}, Error {error_operator} {error:.3E}"
BAYES_SWITCH_TO_SCIENTIFIC = 1e4

WILCOXON_CLAUSE = 'Z(df:.0f) = {statistic:.2f}, ' + P_VALUE_CLAUSE



# TODO - consider removing this class as it doesn't seem that we need the reporter
#  objcet. Everything here can be static.
class Reporter:

    def __init__(self):
        pass

    def visit(self, model):
        if isinstance(model, groupwise.T1SampleModel):
            return self._visit_t1_sample(model)

    def _visit_t1_sample(self, model):
        t_dict = model.get_df().to_dict('records')
        t_dict['p_operator'] = '<' if t_dict['p.value'] < .001 else '='
        return FREQ_T_CLAUSE.format(**t_dict)
        # t, df, p, d, 95% ci, tail

    def _visit_anova(self, model):
        anova_terms = model.get_df().to_dict('records')
        return '. '.join([F_CLAUSE.format(**f) for f in anova_terms])

    def _visit_bayesian_t_test(self, model):
        b_dict = model.get_df().to_dict('records')
        b_dict['error_operator'] = '=' if b_dict['error'] > 0.001 else '<'
        if np.any(np.array(
                [b_dict['bf'], 1 / b_dict['bf']]) > BAYES_SWITCH_TO_SCIENTIFIC):
            return BAYES_T_CLAUSE_SCI_NOTATION.format(**b_dict)
        return BAYES_T_CLAUSE_DEC_NOTATION.format(**b_dict)

    def _visit_t1_sample(self, model):
        t_dict = model.get_df().to_dict('records')
        t_dict['p_operator'] = '<' if t_dict['p.value'] < .001 else '='
        return FREQ_T_CLAUSE.format(**t_dict)



# class _TextualReporter:
#
#     def __init__(self, data, analysis_type, **kwargs) -> None:
#         self._verify_labels(data, analysis_type)
#
#     def process_frequentist_term(self, term_dict: dict,
#                                  test_type: str = 'hypothesis',
#                                  term_type: str = ''):
#         """
#
#         @type test_type: object
#         @param test_type:
#         @type d: object
#         """
#         f" {self._process_significance_result(data['p.value'])} significant effect between"
#         result_clause = (f"[\u03C7\u00B2({term_dict['parameter']:.0f})" +
#                          f" = {term_dict['statistic']:.2f}, p = {term_dict['p.value']:.3f}]")
#
#     def _process_significance_result(self, sig, alpha=.05):
#         return np.where(sig < alpha, 'a', 'no')
#
#     def _verify_labels(self, data):
#         pass
