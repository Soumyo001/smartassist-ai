
"""SQLite storage for SmartAssist AI's fictional online store."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "smartassist.db"


def get_connection():
    """Open a database connection with foreign-key checks enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database():
    """Create tables and insert demo records without duplicating existing data."""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                price_bdt INTEGER NOT NULL CHECK (price_bdt >= 0),
                stock INTEGER NOT NULL CHECK (stock >= 0)
            );

            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_number TEXT NOT NULL UNIQUE,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                status TEXT NOT NULL CHECK (
                    status IN ('pending', 'confirmed', 'shipped', 'delivered', 'cancelled')
                ),
                total_bdt INTEGER NOT NULL CHECK (total_bdt >= 0),
                shipping_address TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL REFERENCES orders(id),
                product_id INTEGER NOT NULL REFERENCES products(id),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                unit_price_bdt INTEGER NOT NULL CHECK (unit_price_bdt >= 0),
                UNIQUE (order_id, product_id)
            );

            CREATE TABLE IF NOT EXISTS support_tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                order_id INTEGER REFERENCES orders(id),
                subject TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open' CHECK (
                    status IN ('open', 'in_progress', 'resolved')
                ),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)

        products = [
            ("P001", "Wireless Headphones", "Electronics", "Over-ear Bluetooth headphones", 2800, 12),
            ("P002", "Gaming Mouse", "Accessories", "Wired mouse with adjustable DPI", 1450, 8),
            ("P003", "USB-C Charger", "Accessories", "30W USB-C fast charger", 950, 0),
            ("P004", "Bluetooth Speaker", "Electronics", "Portable wireless speaker", 3200, 5),
            ("P005", "Wireless Earbuds", "Electronics", "Compact wireless earbuds", 2400, 14),
            ("P006", "Laptop Stand", "Accessories", "Adjustable aluminum laptop stand", 1800, 9),
        ]
        conn.executemany("""
            INSERT OR IGNORE INTO products
            (sku, name, category, description, price_bdt, stock)
            VALUES (?, ?, ?, ?, ?, ?)
        """, products)

        customers = [
            ("Amina Ahmed", "amina@example.com"),
            ("Rahim Hasan", "rahim@example.com"),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO customers (name, email) VALUES (?, ?)",
            customers,
        )

        sample_orders = [
            ("SA-1001", "amina@example.com", "P001", 1, "delivered", "10 Demo Road, Dhaka"),
            ("SA-1002", "rahim@example.com", "P006", 2, "shipped", "25 Example Lane, Dhaka"),
        ]
        for number, email, sku, quantity, status, address in sample_orders:
            customer = conn.execute(
                "SELECT id FROM customers WHERE email = ?", (email,)
            ).fetchone()
            product = conn.execute(
                "SELECT id, price_bdt FROM products WHERE sku = ?", (sku,)
            ).fetchone()

            conn.execute("""
                INSERT OR IGNORE INTO orders
                (order_number, customer_id, status, total_bdt, shipping_address)
                VALUES (?, ?, ?, ?, ?)
            """, (number, customer["id"], status, product["price_bdt"] * quantity, address))

            order = conn.execute(
                "SELECT id FROM orders WHERE order_number = ?", (number,)
            ).fetchone()
            conn.execute("""
                INSERT OR IGNORE INTO order_items
                (order_id, product_id, quantity, unit_price_bdt)
                VALUES (?, ?, ?, ?)
            """, (order["id"], product["id"], quantity, product["price_bdt"]))

        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    initialize_database()
    conn = get_connection()
    try:
        print("Database ready: smartassist.db")
        for table in ("products", "customers", "orders", "order_items", "support_tickets"):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table}: {count}")
        print("Sample orders:")
        for row in conn.execute("SELECT order_number, status, total_bdt FROM orders ORDER BY order_number"):
            print(f"  {row['order_number']} | {row['status']} | BDT {row['total_bdt']}")
    finally:
        conn.close()
