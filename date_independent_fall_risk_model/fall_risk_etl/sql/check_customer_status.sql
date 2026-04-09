SELECT
    account_id,
    account_number,
    account_record_type_name,
    acquisition_partner_name,
    account_status,
    cancel_requested_on,
    account_status_detail,
    account_created_on
FROM DATAHUB.analytics.dim_core__sf_account_customer_details