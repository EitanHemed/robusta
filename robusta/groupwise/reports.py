import numpy as np

from . import models

P_VALUE_CLAUSE = 'p {pvalue_operator} {p-value:.3f}'

FREQ_T_CLAUSE = (
        't({df:.0f}) = {t:.2f}, ' + P_VALUE_CLAUSE)
COHEN_D_CLAUSE = "Cohen's d = {Cohen-d:.2f}"
COHEN_D_INTERVAL_CLAUSE = '({Cohen-d Low:.2f}, {Cohen-d High:.2f})'

BAYES_T_CLAUSE_DEC_NOTATION = ('{model} [BF1:0 = {bf:.2f}, Error = {error:.3f}%]')
BAYES_T_CLAUSE_SCI_NOTATION = ('{model} [BF1:0 = {bf:.2E}, Error = {error:.3f}]')
BAYES_SWITCH_TO_SCIENTIFIC = 1e4

WILCOXON_CLAUSE = 'Z = {Z:.2f}, ' + P_VALUE_CLAUSE

# TODO Currently the class would always report ANOVA effect size using PES rather than an option to get GES.
GENERAL_F_CLAUSE = ('{Term} [F({df1:.0f}, {df2:.0f}) = {F:.2f}, ')
ETA_SQ_CLAUSE = ', Partial Eta-Sq. = {Partial Eta-Squared:.2f}'
ANOVA_TERM_CLAUSE = (GENERAL_F_CLAUSE + P_VALUE_CLAUSE + ETA_SQ_CLAUSE + ']')
ART_TERM_CLAUSE = (GENERAL_F_CLAUSE + P_VALUE_CLAUSE + ']')

BAYES_ANOVA_TERM_CLAUSE_DEC_NOTATION = 'model - ' + BAYES_T_CLAUSE_DEC_NOTATION
BAYES_ANOVA_TERM_CLAUSE_SCI_NOTATION = 'model - ' + BAYES_T_CLAUSE_SCI_NOTATION

KRUSKSAL_WALLIS_CLAUSE = ('H({df1:.0f}, {df2:.0f}) = {statistic:.2f}, ' + P_VALUE_CLAUSE)
FRIEDMAN_CLAUSE = ('Z({df1:.0f}, {df2:.0f}) = {statistic:.2f}, ' + P_VALUE_CLAUSE)


# TODO - consider removing this class as it doesn't seem that we need the reporter
#  objcet. Everything here can be static.


class Reporter:

    def __init__(self):
        pass

    def report_table(self, model):
        return model._get_r_output_df()

    def report_text(self, model, as_list=False, effect_size=False, es=False):

        # Alias effect size
        effect_size = es if es else effect_size

        # Similar reports on all Bayesian Models
        if isinstance(model,
                      (models.BayesT1Sample,
                       models.BayesT2Samples,
                       models.BayesAnova
                       )):
            return self._populate_bayes_t_test_clause(model)

        # Frequentist Models (these are children classes, so they are placed first)
        elif isinstance(model, (models.Wilcoxon1Sample, models.Wilcoxon2Samples)):
            return self._populate_frequentist_clause(model, WILCOXON_CLAUSE)
        elif isinstance(model, models.KruskalWallisTest):
            return self._populate_frequentist_clause(model, KRUSKSAL_WALLIS_CLAUSE)
        elif isinstance(model, models.FriedmanTest):
            return self._populate_frequentist_clause(model, FRIEDMAN_CLAUSE)
        # Frequentist Models (these are mostly parent classes, so they are tested for later)
        elif isinstance(model, (models.T1Sample, models.T2Samples)):
            _effect_size_clause = f', {COHEN_D_CLAUSE}, {COHEN_D_INTERVAL_CLAUSE}' if effect_size else ''
            return self._populate_frequentist_clause(model, FREQ_T_CLAUSE, effect_size_clause=_effect_size_clause)

        # Frequentist Models (these are mostly parent classes, so they are tested for later)
        elif isinstance(model, models.AlignedRanksTest):
            return self._populate_frequentist_clause(model, ART_TERM_CLAUSE, as_list=as_list)
        elif isinstance(model, models.Anova):
            return self._populate_frequentist_clause(model, ANOVA_TERM_CLAUSE, as_list=as_list)
        else:
            raise NotImplementedError

    def _populate_frequentist_clause(self, model, clause, as_list=False, effect_size_clause=''):
        df = model.report_table()
        df['pvalue_operator'] = np.where(df['p-value'] < 0.001, '<', '=')
        df.loc[df['p-value'] < 0.001, 'p-value'] = 0.001
        terms = [(clause+effect_size_clause).format(**f) for f in df.to_dict('records')]
        if as_list:
            return terms
        return '. '.join(terms)

    def _populate_bayes_t_test_clause(self, model, as_list=False):

        df = model.report_table()

        df['error_operator'] = np.where(df['error'] < 0.001, '<', '=')
        df.loc[df['error'] < 0.001, 'error'] = 0.001

        if np.any(np.array(
                [df['bf'], 1 / df['bf']]) > BAYES_SWITCH_TO_SCIENTIFIC):
            clause = BAYES_T_CLAUSE_SCI_NOTATION
        else:
            clause = BAYES_T_CLAUSE_DEC_NOTATION

        terms = [clause.format(**f) for f in df.to_dict('records')]

        if as_list:
            return terms
        return '. '.join(terms)

    # def _populate_wilcoxon_test_clause(self, model):
    #     w_dict = model.report_table().to_dict('records')[0]
    #     w_dict['p_operator'] = '<' if w_dict['p.value'] < .001 else '='
    #     return WILCOXON_CLAUSE.format(**w_dict)

    # def _populate_anova_like_clauses(self, model, as_list, clause=ANOVA_TERM_CLAUSE):
    #     df = model.report_table()
    #     df['pvalue_operator'] = np.where(df['p-value'] < 0.001, '<', '=')
    #     anova_terms = df.to_dict('records')
    #     terms = [clause.format(**f) for f in anova_terms]
    #
    #     if as_list:
    #         return terms
    #     return '. '.join(terms)

    # def _populate_bayes_anova_clauses(self, model, as_list):
    #     bayes_terms = model.report_table().to_dict('records')
    #     terms = []
    #     for b_dict in bayes_terms:
    #         b_dict['error_operator'] = '=' if b_dict['error'] > 0.001 else '<'
    #         if np.any(np.array(
    #                 [b_dict['bf'],
    #                  1 / b_dict['bf']]) > BAYES_SWITCH_TO_SCIENTIFIC):
    #             terms.append(
    #                 BAYES_ANOVA_TERM_CLAUSE_SCI_NOTATION.format(**b_dict))
    #         terms.append(BAYES_ANOVA_TERM_CLAUSE_DEC_NOTATION.format(**b_dict))
    #     if as_list:
    #         return terms
    #     return '. '.join(terms)
