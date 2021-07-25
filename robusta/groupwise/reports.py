import numpy as np

from . import models

P_VALUE_CLAUSE = 'p {pvalue_operator} {p-value:.3f}'

FREQ_T_CLAUSE = (
        't({dof:.0f}) = {t:.2f}, ' + P_VALUE_CLAUSE)
COHEN_D_CLAUSE = "Cohen's d = {cohen_d:.2f}"
COHEN_D_INTERVAL_CLAUSE = '({low:.2f, high:.2f})'

# FREQ_T_DIFFERENCE_CLAUSE = "Mean Difference = {mean_diff:.2f}"

BAYES_T_CLAUSE_DEC_NOTATION = ('BF1:0 = {bf:.2f}, '
                               'Error {error_operator} {error:.3f}%')
BAYES_T_CLAUSE_SCI_NOTATION = ('BF1:0 = {bf:.2E}, '
                               'Error {error_operator} {error:.3f}')
BAYES_SWITCH_TO_SCIENTIFIC = 1e4

WILCOXON_CLAUSE = 'Z(df:.0f) = {statistic:.2f}, ' + P_VALUE_CLAUSE

ANOVA_TERM_CLAUSE = ('{Term}'
                     ' ['
                     'F({df1:.0f}, '
                     '{df2:.0f}) = {F:.2f}, '
                     + P_VALUE_CLAUSE
                     + ', Partial Eta-Sq. = {Partial Eta-Squared:.2f}'
                     ']')
BAYES_ANOVA_TERM_CLAUSE_DEC_NOTATION = 'model - ' + BAYES_T_CLAUSE_DEC_NOTATION
BAYES_ANOVA_TERM_CLAUSE_SCI_NOTATION = 'model - ' + BAYES_T_CLAUSE_SCI_NOTATION

KRUSKSAL_WALLIS_CLAUSE = ('H({df1:.0f}, '
                          '{df2:.0f}) = {statistic:.2f}, ' + P_VALUE_CLAUSE)
FRIEDMAN_CLAUSE = ('Z({df1:.0f}, '
                   '{df2:.0f}) = {statistic:.2f}, ' + P_VALUE_CLAUSE)


# # Due to a circular import we have to put this below the top level.
# if __name__ == '__main__':
#     FREQUENTIST_ANOVA_LIKE_CLAUSE_DICT = {
#         models.AlignedRanksTest: ANOVA_TERM_CLAUSE,
#         models.Anova: ANOVA_TERM_CLAUSE,
#         models.FriedmanTest: FRIEDMAN_CLAUSE,
#         models.KruskalWallisTest: KRUSKSAL_WALLIS_CLAUSE
#     }


# TODO - consider removing this class as it doesn't seem that we need the reporter
#  objcet. Everything here can be static.
class Reporter:

    def __init__(self):
        pass

    def report_table(self, model):
        return model._get_r_output_df()

    def report_text(self, model):
        if isinstance(model,
                      (models.BayesT1Sample,
                       models.BayesT2Samples)):
            return self._populate_bayes_t_test_clause(model)
        elif isinstance(model,
                        (models.T1Sample,
                         models.T2Samples)):
            return self._populate_t_test_clause(model)
        elif isinstance(model, models.KruskalWallisTest):
            return self._populate_anova_like_clauses(model, KRUSKSAL_WALLIS_CLAUSE)
        elif isinstance(model, models.FriedmanTest):
            return self._populate_anova_like_clauses(model, FRIEDMAN_CLAUSE)
        elif isinstance(model, models.BayesAnova):
            return self._populate_bayes_anova_clauses(model)
        elif isinstance(model, models.Anova):
            return self._populate_anova_like_clauses(model)
        else:
            raise NotImplementedError

    def _populate_t_test_clause(self, model):
        t_dict = model.report_table().to_dict('records')[0]
        t_dict['p_operator'] = '<' if t_dict['p.value'] < .001 else '='
        return FREQ_T_CLAUSE.format(**t_dict)

    def _populate_bayes_t_test_clause(self, model):
        b_dict = model.report_table().to_dict('records')[0]
        b_dict['error_operator'] = '=' if b_dict['error'] > 0.001 else '<'
        if np.any(np.array(
                [b_dict['bf'], 1 / b_dict['bf']]) > BAYES_SWITCH_TO_SCIENTIFIC):
            return BAYES_T_CLAUSE_SCI_NOTATION.format(**b_dict)
        return BAYES_T_CLAUSE_DEC_NOTATION.format(**b_dict)

    def _populate_wilcoxon_test_clause(self, model):
        w_dict = model.report_table().to_dict('records')[0]
        w_dict['p_operator'] = '<' if w_dict['p.value'] < .001 else '='
        return WILCOXON_CLAUSE.format(**w_dict)

    def _populate_anova_like_clauses(self, model, clause=ANOVA_TERM_CLAUSE):
        df = model.report_table()
        df['pvalue_operator'] = np.where(df['p-value'] < 0.001, '<', '=')
        anova_terms = df.to_dict('records')
        return '. '.join([clause.format(**f) for f in anova_terms])

    def _populate_bayes_anova_clauses(self, model):
        bayes_terms = model.report_table().to_dict('records')
        terms = []
        for b_dict in bayes_terms:
            b_dict['error_operator'] = '=' if b_dict['error'] > 0.001 else '<'
            if np.any(np.array(
                    [b_dict['bf'],
                     1 / b_dict['bf']]) > BAYES_SWITCH_TO_SCIENTIFIC):
                terms.append(
                    BAYES_ANOVA_TERM_CLAUSE_SCI_NOTATION.format(**b_dict))
            terms.append(BAYES_ANOVA_TERM_CLAUSE_DEC_NOTATION.format(**b_dict))
        return '. '.join(terms)
