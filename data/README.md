# Data

This directory contains raw and processed transaction data (gitignored).

## Data Sources
- eBay Seller Hub transaction export (CSV)
- Date range: January 2025 - June 2026
- Store: Vintage sampling media resale business

## Files
- `ebay_transactions.csv` — raw eBay export (not tracked)
- `ebay_orders_clean.csv` — cleaned order-level data (not tracked)

## Schema
- buyer_username — anonymized eBay buyer ID
- order_date — transaction date
- order_value — sum of item prices per order
- buyer_state — US state (domestic orders)
- buyer_country — buyer country