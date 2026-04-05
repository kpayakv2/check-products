-- Fix product_id to be nullable in product_category_suggestions
-- This allows suggestions to be created without linking to a specific product initially

ALTER TABLE product_category_suggestions 
ALTER COLUMN product_id DROP NOT NULL;

-- Also fix product_attributes table
ALTER TABLE product_attributes 
ALTER COLUMN product_id DROP NOT NULL;

-- Add comment to clarify the purpose
COMMENT ON COLUMN product_category_suggestions.product_id IS 'Optional reference to products table. Can be NULL for standalone suggestions.';
COMMENT ON COLUMN product_attributes.product_id IS 'Optional reference to products table. Can be NULL for standalone attributes.';
