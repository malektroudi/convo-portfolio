-- Response latency, as percentiles rather than a mean.
--
-- A mean latency is the number that hides the problem: it is dragged by the
-- tail it is meant to expose, and no request ever took it. The percentile a
-- product owner should be shown is p90, because that is roughly the
-- experience of the unluckiest caller in every ten.
--
-- Rows with no measured latency are excluded rather than zero-filled. Day 10
-- and day 11 never timed their turns, and a null is the honest record of
-- that; a zero would quietly improve every figure here.
SELECT
  surface,
  COUNT(*)                                   AS timed_turns,
  ROUND(AVG(latency_ms))                     AS mean_ms,
  ROUND(APPROX_QUANTILE(latency_ms, 0.5))    AS p50_ms,
  ROUND(APPROX_QUANTILE(latency_ms, 0.9))    AS p90_ms,
  ROUND(MAX(latency_ms))                     AS worst_ms
FROM {table}
WHERE latency_ms IS NOT NULL
GROUP BY surface
ORDER BY timed_turns DESC
