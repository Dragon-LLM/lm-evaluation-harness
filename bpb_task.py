import math
from lm_eval.api.task import ConfigurableTask


def mean(items):
    return sum(float(x) for x in items) / max(1, len(items))


class BPBConfigurableTask(ConfigurableTask):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

    def construct_requests(self, doc, ctx, **kwargs):
        return super().construct_requests(doc, ctx, **kwargs)

    def _extract_target(self, doc):
        try:
            target = self.doc_to_target(doc)
        except Exception:
            target = None

        if not target:
            target = (
                doc.get("canonical_solution")
                or doc.get("code")
                or doc.get("solution")
                or doc.get("answer")
                or doc.get("target")
                or ""
            )

        if not isinstance(target, str):
            target = str(target)

        return target

    def _extract_loglikelihood(self, results):
        first = results[0] if isinstance(results, list) and len(results) > 0 else results
        if isinstance(first, tuple):
            return float(first[0])
        return float(first)

    def process_results(self, doc, results):
        target = self._extract_target(doc)
        byte_len = max(1, len(target.encode("utf-8")))

        loglik = self._extract_loglikelihood(results)

        bits_total = -loglik / math.log(2)
        bpb = bits_total / byte_len

        return {
            "bpb": bpb,
            "bits_total": bits_total,
            "bytes_total": float(byte_len),
        }

    def aggregation(self):
        return {
            "bpb": mean,
            "bits_total": sum,
            "bytes_total": sum,
        }

    def higher_is_better(self):
        return {
            "bpb": False,
            "bits_total": False,
            "bytes_total": False,
        }