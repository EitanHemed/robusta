import typing
import warnings
import numpy as np
from .. import pyr
from ..misc import utils, base

BF_COLUMNS = ['model', 'bf', 'error']
REDUNDANT_BF_COLUMNS = ['time', 'code']

DEFAULT_GROUPWISE_NULL_INTERVAL = (-np.inf, np.inf)
PRIOR_SCALE_STR_OPTIONS = ('medium', 'wide', 'ultrawide')
DEFAULT_ITERATIONS: int = 10000
DEFAULT_MU: float = 0.0

T_TEST_COLUMNS_RENAME = {'statistic': 't', 'parameter': 'dof', 'estimate': 'mean', 'p.value': 'p-value'}
T_TEST_COHEN_COLUMN_NAMES = ['Cohen-d Low', 'Cohen-d', 'Cohen-d High']
T_TEST_RETURNED_COLUMNS = [T_TEST_COLUMNS_RENAME['statistic'], T_TEST_COLUMNS_RENAME['parameter'],
                           T_TEST_COLUMNS_RENAME['p.value'], 'Cohen-d', 'Cohen-d Low', 'Cohen-d High']

ANOVA_COLUMNS_RENAME = {'Effect':'Term', 'p.value': 'p-value'}

class GroupwiseResults(base.BaseResults):

    def get_text(self, mode: str = 'df'):
        raise NotImplementedError


class T2SamplesResults(GroupwiseResults):

    def _reformat_r_output_df(self):
        df = self._get_r_output_df().copy()
        df.rename(columns=T_TEST_COLUMNS_RENAME, inplace=True)

        vals = df.to_dict('records')[0]

        dofs = vals['df']

        # TODO - how to handle unequal group size
        cohen_d = pyr.rpackages.psych.t2d(vals['t'], n=vals['dof'])
        cohen_d_ci = pyr.rpackages.psych.d_ci(cohen_d, n=vals['dof'])

        df[T_TEST_COHEN_COLUMN_NAMES] = cohen_d_ci  # Skip the middle value

        return df[T_TEST_RETURNED_COLUMNS]


class BayesT2SamplesResults(T2SamplesResults):
    # TODO Validate if correctly inherits init from parent

    def __init__(self, r_results, mode='bf'):
        self.mode = mode

        super().__init__(r_results)

    def _tidy_results(self):
        if self.mode == 'bf':
            return utils.convert_df(self.r_results, 'model')[BF_COLUMNS]
        else:
            return utils.convert_df(self.r_results)

    def _get_r_output_df(self):
        return self._tidy_results()

    def _reformat_r_output_df(self):
        return self.get_df()

    def get_df(self):
        return self._get_r_output_df()


class T1SampleResults(T2SamplesResults):
    pass


# TODO - this is fairly similar to the two-sample bayesian t-test, so consider
#  merging them
class BayesT1SampleResults(T1SampleResults):

    def __init__(self, r_results, mode='bf'):
        self.mode = mode
        super().__init__(r_results)

    def _tidy_results(self):
        if self.mode == 'bf':
            return utils.convert_df(self.r_results, 'model')[BF_COLUMNS]
        else:
            return utils.convert_df(self.r_results, 'iteration')

    def _get_r_output_df(self):
        return self._tidy_results()


class Wilcoxon1SampleResults(T1SampleResults):
    pass


class Wilcoxon2SamplesResults(T2SamplesResults):
    pass


class AnovaResults(GroupwiseResults):

    def _tidy_results(self):
        # TODO - this is an ugly hack. The problem is that up until now we didn't have the auto convertsion to recarray
        #  that removes the rownames. This needs to be fixed.
        return pyr.rpackages.afex.nice(self.r_results)

        # anova_table = utils.convert_df(
        #     pyr.rpackages.stats.anova(
        #         self.r_results))
        # missing_column = utils.convert_df(pyr.rpackages.afex.nice(self.r_results))['Effect'].values
        # anova_table.insert(loc=0, column='term', value=missing_column)
        # return anova_table

    def _reformat_r_output_df(self):
        df = self._get_r_output_df().copy()
        df.rename(columns=ANOVA_COLUMNS_RENAME, inplace=True)



        # TODO - how to handle unequal group size
        cohen_d = pyr.rpackages.psych.t2d(vals['t'], n=vals['dof'])
        cohen_d_ci = pyr.rpackages.psych.d_ci(cohen_d, n=vals['dof'])

        df[T_TEST_COHEN_COLUMN_NAMES] = cohen_d_ci  # Skip the middle value

        return df[T_TEST_RETURNED_COLUMNS]

    def get_margins(
            self,
            margins_terms: typing.List[str] = None,
            by_terms: typing.List[str] = None,
            ci: int = 95,
            overwrite: bool = False
    ):
        # TODO Look at this - emmeans::as_data_frame_emm_list
        # TODO - Documentation
        # TODO - Issue the 'balanced-design' warning etc.
        # TODO - allow for `by` option
        # TODO - implement between and within CI calculation

        _terms_to_test = np.array(
            utils.to_list([margins_terms,
                           [] if by_terms is None else by_terms])
        )

        if not all(
                term in self._get_r_output_df()['Effect'].values for term in
                _terms_to_test):
            raise RuntimeError(
                f'Margins term: {[i for i in _terms_to_test]}'
                'not included in model')

        margins_terms = np.array(utils.to_list(margins_terms))

        if by_terms is None:
            by_terms = pyr.rinterface.NULL
        else:
            by_terms = np.array(utils.to_list(by_terms))

        _r_margins = pyr.rpackages.emmeans.emmeans(
            self.r_results,
            specs=margins_terms,
            type='response',
            level=ci,
            by=by_terms
        )

        margins = utils.convert_df(
            pyr.rpackages.emmeans.as_data_frame_emmGrid(
                _r_margins))

        return margins

        # self.margins_results[tuple(margins_term)] = {
        #    'margins': margins, 'r_margins': _r_margins}

        # if self.margins_results is not None and overwrite is False:
        #     raise RuntimeError(
        #         'Margins already defined. To re-run, get_margins'
        #         'with `overwrite` kwarg set to True')
        #
        # if margins_term is None:
        #     if self.margins_term is None:
        #         raise RuntimeError('No margins term defined')
        # else:
        #     self.margins_term = margins_term
        #
        #
        # # TODO Currently this does not support integer arguments if we would
        # #  get those. Also we need to make sure that we get a list or array
        # #  as RPy2 doesn't take tuples. Probably should go with array as this
        # #  might save time on checking whether the margins term is in the model,
        # #  without making sure we are not comparing a string and a list.
        # if isinstance(margins_term, str):
        #     margins_term = [margins_term]
        # margins_term = np.array(margins_term)
        # if not all(
        #         term in self.get_df()['term'].values for term in
        #         margins_term):
        #     raise RuntimeError(
        #         f'Margins term: {[i for i in margins_term]}'
        #         'not included in model')
        #
        # _r_margins = pyr.rpackages.emmeans.emmeans(
        #     self.r_results,
        #     specs=margins_term,
        #     type='response',
        #     level=ci,
        # )
        # margins = utils.convert_df(
        #     pyr.rpackages.emmeans.as_data_frame_emmGrid(
        #         _r_margins))
        #
        # self.margins_results[tuple(margins_term)] = {
        #     'margins': margins, 'r_margins': _r_margins}
        # return margins

    def get_pairwise(self,
                     margins_term: typing.Optional[typing.Union[str]] = None,
                     overwrite_pairwise_results: bool = False):
        # TODO - Documentation.
        # TODO - Testing.
        # TODO - implement pairwise options (e.g., `infer`)
        warnings.warn('Currently under development. Expect bugs.')

        if not isinstance(str, margins_term):
            if self.margins_term is None:
                raise RuntimeError("No margins_term defined")
            else:
                margins_term = self.margins_term
        if '*' in margins_term:  # an interaction
            raise RuntimeError(
                'get_pairwise cannot be run using an interaction'
                'margins term')
        if margins_term in self.margins_results:
            if ('pairwise' in self.margins_results[margins_term]
                    and not overwrite_pairwise_results):
                raise RuntimeError(
                    'Margins already defined. To re-run, get_margins'
                    'with `overwrite_margins_results` kwarg set '
                    'to True')
        return utils.convert_df(pyr.rpackages.emmeans.pairs(
            self.margins_results[margins_term]['r_margins']))


class BayesAnovaResults(AnovaResults):

    def _tidy_results(self):
        return utils.convert_df(
            self.r_results, 'model')

    def _get_r_output_df(self):
        return self._tidy_results()[['model', 'bf', 'error']]

    def get_margins(self):
        raise NotImplementedError("Not applicable to Bayesian ANOVA")


class KruskalWallisTestResults(AnovaResults):

    def _tidy_results(self):
        return pyr.rpackages.generics.tidy(self.r_results)

    def get_margins(self):
        raise NotImplementedError("Not applicable to non-parametric ANOVA")


class FriedmanTestResults(AnovaResults):

    def get_margins(self):
        raise NotImplementedError("Not applicable to non-parametric ANOVA")


class AlignedRanksTestResults(AnovaResults):

    def get_margins(self):
        raise NotImplementedError("Not applicable to non-parametric ANOVA")

    def _tidy_results(self):
        return pyr.rpackages.stats.anova(
            self.r_results
        )
