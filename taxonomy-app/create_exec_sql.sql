CREATE OR REPLACE FUNCTION public.exec_sql(
    query_text TEXT,
    query_params JSONB DEFAULT '[]'
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    result JSONB;
    row_count INTEGER;
BEGIN
    -- Execute dynamic SQL
    EXECUTE query_text;
    
    GET DIAGNOSTICS row_count = ROW_COUNT;
    
    -- Return basic execution info
    result = jsonb_build_object(
        'success', true,
        'rows_affected', row_count,
        'executed_at', now()
    );
    
    RETURN result;
EXCEPTION
    WHEN OTHERS THEN
        RETURN jsonb_build_object(
            'success', false,
            'error', SQLERRM,
            'error_code', SQLSTATE
        );
END;
$$;

-- Grant execute permission to service role
GRANT EXECUTE ON FUNCTION public.exec_sql(TEXT, JSONB) TO service_role;
