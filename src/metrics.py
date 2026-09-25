def precision(tp, fp):

    denominator = tp + fp

    if denominator == 0:
        return 0.0

    return tp / denominator


def recall(tp, fn):

    denominator = tp + fn

    if denominator == 0:
        return 0.0

    return tp / denominator


def fbeta(
    tp,
    fp,
    fn,
    beta=0.5,
):

    p = precision(tp, fp)
    r = recall(tp, fn)

    if p == 0 and r == 0:
        return 0.0

    beta_squared = beta ** 2

    return (
        (1 + beta_squared)
        * p
        * r
    ) / (
        beta_squared * p
        + r
    )


def f05(
    tp,
    fp,
    fn,
):
    return fbeta(
        tp,
        fp,
        fn,
        beta=0.5,
    )
