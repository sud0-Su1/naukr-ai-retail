from pathlib import Path
import pandas as pd


OUTPUT_DIR = Path(__file__).parent


def create_products() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "product_id": "P001",
            "sku": " sku-001 ",
            "product_name": "Classic T-Shirt",
            "category": "T-Shirt",
            "brand": " Nike",
            "price": "1299",
            "cost": "700",
        },
        {
            "product_id": "P002",
            "sku": "SKU-002",
            "product_name": "Running Shoes",
            "category": " footwear ",
            "brand": "Adidas ",
            "price": "2,499",
            "cost": "1500",
        },
        {
            "product_id": "P003",
            "sku": "sku-003",
            "product_name": "Polo Tee",
            "category": "tee",
            "brand": "nike",
            "price": "999",
            "cost": "500",
        },
        {
            "product_id": "P004",
            "sku": "SKU-004",
            "product_name": "Denim Jeans",
            "category": "Jeans",
            "brand": " Levi's",
            "price": "1999",
            "cost": "1100",
        },
        {
            "product_id": "P005",
            "sku": "SKU-005",
            "product_name": "Hoodie",
            "category": "Hoodies",
            "brand": "Puma",
            "price": "₹1799",
            "cost": "950",
        },
    ])


def create_customers() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "customer_id": "C001",
            "name": "Rahul Sharma ",
            "region": "West",
            "signup_date": "2025-01-10",
        },
        {
            "customer_id": "C002",
            "name": " Priya Singh",
            "region": " west ",
            "signup_date": "10/02/2025",
        },
        {
            "customer_id": "C003",
            "name": "Amit Kumar",
            "region": "NORTH",
            "signup_date": "2025-02-15",
        },
        {
            "customer_id": "C004",
            "name": "Neha Verma",
            "region": "South",
            "signup_date": "03/01/2025",
        },
        {
            "customer_id": "C005",
            "name": "Arjun Mehta",
            "region": "East",
            "signup_date": "2025-03-12",
        },
    ])


def create_stores() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "store_id": "S001",
            "store_name": " Mumbai Central ",
            "region": "West",
            "city": "Mumbai",
        },
        {
            "store_id": "S002",
            "store_name": "Delhi Main",
            "region": "NORTH",
            "city": " Delhi ",
        },
        {
            "store_id": "S003",
            "store_name": "Bangalore Store",
            "region": "South",
            "city": "Bengaluru",
        },
        {
            "store_id": "S004",
            "store_name": "Kolkata Hub",
            "region": "East",
            "city": "Kolkata",
        },
    ])


def create_orders() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "order_id": "ORD001",
            "customer_id": "C001",
            "product_id": "P001",
            "store_id": "S001",
            "order_date": "2025-01-10",
            "quantity": "2",
            "unit_price": "1299",
            "discount": "0.10",
            "returned": "No",
        },
        {
            "order_id": "ORD002",
            "customer_id": "C002",
            "product_id": "P003",
            "store_id": "S001",
            "order_date": "10/02/2025",
            "quantity": "1",
            "unit_price": "999",
            "discount": "0",
            "returned": "Yes",
        },
        {
            "order_id": "ORD003",
            "customer_id": "C003",
            "product_id": "P002",
            "store_id": "S002",
            "order_date": "2025-02-15",
            "quantity": "3",
            "unit_price": "2,499",
            "discount": "0.05",
            "returned": "No",
        },
        {
            "order_id": "ORD004",
            "customer_id": "C004",
            "product_id": "P004",
            "store_id": "S003",
            "order_date": "03/01/2025",
            "quantity": "1",
            "unit_price": "1999",
            "discount": "0.15",
            "returned": "No",
        },
        {
            "order_id": "ORD005",
            "customer_id": "C005",
            "product_id": "P005",
            "store_id": "S004",
            "order_date": "2025-03-12",
            "quantity": "2",
            "unit_price": "₹1799",
            "discount": "0.10",
            "returned": "Yes",
        },
        {
            "order_id": "ORD005",
            "customer_id": "C005",
            "product_id": "P005",
            "store_id": "S004",
            "order_date": "2025-03-12",
            "quantity": "2",
            "unit_price": "₹1799",
            "discount": "0.10",
            "returned": "Yes",
        },
    ])


def create_inventory() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "inventory_id": "I001",
            "product_id": "P001",
            "store_id": "S001",
            "inventory_date": "2025-03-01",
            "stock_qty": "25",
        },
        {
            "inventory_id": "I002",
            "product_id": "P002",
            "store_id": "S002",
            "inventory_date": "01/03/2025",
            "stock_qty": "10",
        },
        {
            "inventory_id": "I003",
            "product_id": "P003",
            "store_id": "S001",
            "inventory_date": "2025-03-01",
            "stock_qty": "0",
        },
        {
            "inventory_id": "I004",
            "product_id": "P004",
            "store_id": "S003",
            "inventory_date": "2025-03-01",
            "stock_qty": "-2",
        },
        {
            "inventory_id": "I005",
            "product_id": "P005",
            "store_id": "S004",
            "inventory_date": "2025-03-01",
            "stock_qty": None,
        },
    ])


def save_dataset(name: str, df: pd.DataFrame) -> None:
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    print(f"Created {path}")


def main() -> None:
    save_dataset("products.csv", create_products())
    save_dataset("customers.csv", create_customers())
    save_dataset("stores.csv", create_stores())
    save_dataset("orders.csv", create_orders())
    save_dataset("inventory.csv", create_inventory())

    print("\nAll sample datasets created successfully.")


if __name__ == "__main__":
    main()