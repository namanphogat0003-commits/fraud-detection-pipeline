-- Run these against data/processed/fraud.db (e.g. with "DB Browser for SQLite",
-- or `sqlite3 data/processed/fraud.db`) after running data_prep.py.
-- This is your SQL practice for the project — know these numbers before you
-- start feature engineering in Week 2.

-- 1. Overall class imbalance
SELECT COUNT(*) AS total,
       SUM(isFraud) AS fraud_count,
       ROUND(100.0 * SUM(isFraud) / COUNT(*), 4) AS fraud_pct
FROM transactions;

-- 2. Fraud rate by transaction type
SELECT type,
       COUNT(*) AS total_txns,
       SUM(isFraud) AS fraud_txns,
       ROUND(100.0 * SUM(isFraud) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY type
ORDER BY fraud_rate_pct DESC;

-- 3. Average / max transaction amount: fraud vs legitimate
SELECT isFraud,
       ROUND(AVG(amount), 2) AS avg_amount,
       ROUND(MAX(amount), 2) AS max_amount
FROM transactions
GROUP BY isFraud;

-- 4. Balance-mismatch signal — a known strong fraud indicator in this dataset.
--    When oldbalanceOrg - amount != newbalanceOrig, the sender's balance
--    didn't reconcile the way it should have.
SELECT nameOrig, amount, oldbalanceOrg, newbalanceOrig, isFraud
FROM transactions
WHERE (oldbalanceOrg - amount) != newbalanceOrig
  AND type IN ('TRANSFER', 'CASH_OUT')
LIMIT 20;

-- 5. isFlaggedFraud vs isFraud — how good is PaySim's own naive rule?
--    (Useful as your "baseline to beat" business comparison later.)
SELECT isFraud, isFlaggedFraud, COUNT(*) AS n
FROM transactions
GROUP BY isFraud, isFlaggedFraud;
