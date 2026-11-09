-- PostgreSQL example: training/inference features as known at :as_of_timestamp.
-- This is a DESIGN EXAMPLE; not executable against IBM's one-table CSV without
-- the event tables documented below. No future churn outcome is used as a feature.
-- Assumed tables:
--   customers(customer_id, signup_at, contract_type)
--   monthly_billing(customer_id, billed_at, charge_usd)
--   service_events(customer_id, event_at, event_type)
-- The query demonstrates JOIN, multiple CTEs and a window function.
WITH previous_bills AS (
    SELECT customer_id, billed_at, charge_usd
    FROM monthly_billing
    WHERE billed_at < :as_of_timestamp  -- strict point-in-time cutoff
),
billing_ranked AS (
    SELECT
        customer_id,
        billed_at,
        charge_usd,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id ORDER BY billed_at DESC
        ) AS bill_rank
    FROM previous_bills
),
billing_summary AS (
    SELECT
        customer_id,
        COUNT(*) AS number_of_prior_bills,
        COALESCE(SUM(charge_usd), 0) AS total_charges_to_date,
        AVG(charge_usd) AS average_monthly_charge
    FROM previous_bills
    GROUP BY customer_id
),
prior_service_events AS (
    SELECT customer_id,
           COUNT(*) FILTER (WHERE event_type = 'support_ticket') AS prior_support_tickets
    FROM service_events
    WHERE event_at < :as_of_timestamp
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    GREATEST(0, EXTRACT(YEAR FROM AGE(:as_of_timestamp, c.signup_at)) * 12
        + EXTRACT(MONTH FROM AGE(:as_of_timestamp, c.signup_at))) AS tenure_months,
    c.contract_type,
    COALESCE(b.number_of_prior_bills, 0) AS number_of_prior_bills,
    COALESCE(b.total_charges_to_date, 0) AS total_charges_to_date,
    b.average_monthly_charge,
    r.charge_usd AS latest_monthly_charge,
    COALESCE(e.prior_support_tickets, 0) AS prior_support_tickets
FROM customers AS c
LEFT JOIN billing_summary AS b USING (customer_id)
LEFT JOIN billing_ranked AS r ON r.customer_id = c.customer_id AND r.bill_rank = 1
LEFT JOIN prior_service_events AS e USING (customer_id)
WHERE c.signup_at < :as_of_timestamp;
