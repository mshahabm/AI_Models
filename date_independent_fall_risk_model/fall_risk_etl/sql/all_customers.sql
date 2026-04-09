SELECT
	account_id,
	account_number,
	age,
	account_record_type_name,
	account_status
FROM
	datahub.analytics.dim_core__sf_account_customer_details
WHERE
	account_record_type_name = 'Active Customer'
	AND account_status IN ('Active')