import numpy as np
import robusta as rst

class _TextualReporter:

    def __init__(self, data, analysis_type, **kwargs) -> None:
        self._verify_labels(data, analysis_type)

    def process_frequentist_term(self, term_dict: dict, test_type: str = 'hypothesis',
                                 term_type: str = ''):
        """

        @type test_type: object
        @param test_type:
        @type d: object
        """
        f" {self._process_significance_result(dat['p.value'])} significant effect between"
        result_clause = (f"[\u03C7\u00B2({term_dict['parameter']:.0f})" +
                         f" = {term_dict['statistic']:.2f}, p = {term_dict['p.value']:.3f}]")



    def _process_significance_result(self, sig, alpha=.05):
        return np.where(sig < alpha, 'a', 'no')

    def _verify_labels(self, data):
        pass

class SingleTermReporter(_TextualReporter):
    pass

class MultiTermReporter(_TextualReporter):
    pass

