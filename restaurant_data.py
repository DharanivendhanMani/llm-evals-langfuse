"""Arusuvai Kitchen facts: the single source of truth the assistant's tools read from.

Provenance (keep this honest, the eval dataset's expected outputs depend on it):
  - hours, address, phone: supplied by the owner of this project (Google-listing style, 2026-09-25)
  - delivery fees by radius: supplied by the owner of this project (2026-09-25)
  - menu, services, catering trays: snapshot of arusuvai-kitchen.ca/menu taken 2026-09-25
    (page fetched and summarised by a model, so re-verify prices against the live site)

Deliberately NOT in here (the website / owner did not provide them). The eval dataset's
`unanswerable` cases test that the assistant does not invent any of these:
  allergen info, halal status, vegan/gluten-free labels, spice levels, reservations,
  payment methods, parking, delivery minimum order / time / partner / service beyond 30 miles,
  catering lead time and serving sizes.
"""

from datetime import date

TIMEZONE = "America/Toronto"

NAME = "Arusuvai Kitchen"
ADDRESS = "1067 Dundas St W, Mississauga, ON L5C 1C3, Canada"
PHONE = "+1 437-223-2413"
WEBSITE = "https://arusuvai-kitchen.ca"
MENU_URL = "https://arusuvai-kitchen.ca/menu"
DESCRIPTION = "South Indian cuisine featuring healthy options, fast service"
SERVICES = ["pickup", "delivery", "group orders", "catering"]
MENU_SNAPSHOT_DATE = "2026-09-25"

# 24h "HH:MM" open/close, local time.
HOURS = {
    "Monday": ("12:00", "22:00"),
    "Tuesday": ("12:00", "22:00"),
    "Wednesday": ("12:00", "22:00"),
    "Thursday": ("12:00", "23:00"),
    "Friday": ("12:00", "23:00"),
    "Saturday": ("09:00", "23:00"),
    "Sunday": ("09:00", "23:00"),
}
HOURS_DISCLAIMER = "Hours might differ (e.g. on holidays); call the restaurant to confirm."

SPECIAL_DAYS = {
    date(2026, 9, 30): (
        "National Day for Truth and Reconciliation. The listing shows the regular Wednesday "
        "hours (12 PM - 10 PM) but warns that hours might differ; advise the customer to call to confirm."
    ),
}

CATERING = {
    "trays": "Small, medium and large trays",
    "price_ranges": {
        "curries": "$75-$180",
        "biryanis": "$100-$200",
        "kothu dishes": "$45-$80",
    },
}


DELIVERY = {
    "fees": [
        {"distance_from_restaurant": "within 10 miles", "fee": "$4.99"},
        {"distance_from_restaurant": "10-30 miles", "fee": "$10.99"},
    ],
}


def _m(category, name, price=None, price_text=None, note=None):
    return {"category": category, "name": name, "price": price, "price_text": price_text, "note": note}


MENU = [
    # Appetizers
    _m("Appetizers", "Marina Fish Fry", 19.99),
    _m("Appetizers", "Mushroom 65", 10.99),
    _m("Appetizers", "Mutton Kola Urundai", 15.99),
    _m("Appetizers", "Chicken Lollypop", 15.99),
    _m("Appetizers", "Paneer 65", 10.99),
    _m("Appetizers", "Masala Fries", 4.99),
    _m("Appetizers", "Chicken 65", 12.99),
    _m("Appetizers", "Cauliflower 65", 10.99),
    # Idli & Vadai
    _m("Idli & Vadai", "Rasam Idly", 10.99),
    _m("Idli & Vadai", "Single Plain Idli", 1.25),
    _m("Idli & Vadai", "Medhu Vadai", 8.99),
    _m("Idli & Vadai", "Curd Vadai", 10.99),
    _m("Idli & Vadai", "Sambar Vadai", 10.99),
    _m("Idli & Vadai", "Rasa Vadai", 10.99),
    # Dosai / Uthappam (only some individual prices are published)
    _m("Dosai", "Ghee Masala Dosai", 13.99),
    _m(
        "Dosai", "Dosai varieties", price_text="$11.99-$18.99",
        note="Plain, Masala, Ghee, Butter, Cheese, Onion, Garlic, Podi, Egg, Chicken Keema, Mutton Keema, "
        "Chocolate. Individual prices are not published except Ghee Masala Dosai.",
    ),
    _m("Uthappam", "Onion Tomato Uthappam", 13.99),
    _m(
        "Uthappam", "Uthappam varieties", price_text="$11.99-$17.99",
        note="Plain, Onion, Tomato, Chilli, Cheese, Podi, Ghee, Butter, with protein additions.",
    ),
    # Parotta
    _m("Parotta", "Arapalayam Veg Kothu Parotta", 13.99),
    _m("Parotta", "Arapalayam Egg Kothu Parotta", 14.99),
    _m("Parotta", "Arapalayam Chicken Kothu Parotta", 15.99),
    _m("Parotta", "Arapalayam Mutton Kothu Parotta", 16.99),
    # Combos
    _m("Combos", "Combos", price_text="$6.99-$17.99", note="Idli / dosai / rice / parotta with a curry option."),
    # Eggs
    _m("Eggs", "Plain Omelette", 4.99),
    _m("Eggs", "Masala Omelette", 5.99),
    _m("Eggs", "Chicken Omelette", 7.99),
    _m("Eggs", "Egg Pepper Fry", 12.99),
    _m("Eggs", "Pudukottai Egg Mass", 12.99),
    # Meals (thali)
    _m("Meals", "Vegetarian Meals", 11.99),
    _m("Meals", "Chicken Meals", 13.99),
    _m("Meals", "Fish Meals", 15.99),
    _m("Meals", "Mutton Meals", 15.99),
    _m("Meals", "Prawn Meals", 16.99),
    # Biryani
    _m("Biryani", "Chennai Kalyana Veg Biryani", 12.99),
    _m("Biryani", "Chennai Kalyana Chicken Biryani", 13.99),
    _m("Biryani", "Boneless Chicken 65 Biryani", 15.99),
    _m("Biryani", "Dindigul Mutton Biryani", 16.99, note="Available Wednesdays and weekends only."),
    # Curries
    _m("Curries", "Vegetarian Kurma", 11.99),
    _m("Curries", "Paneer Kurma", 12.99),
    _m("Curries", "Egg Masala", 12.99),
    _m("Curries", "Chicken Curry", 13.99),
    _m("Curries", "Chettinad Chicken", 14.99),
    _m("Curries", "Pepper Chicken", 14.99),
    _m("Curries", "Mutton Curry", 15.99),
    _m("Curries", "Chettinad Mutton", 16.99),
    _m("Curries", "Prawn Masala", 16.99),
    # Dry dishes
    _m("Dry Dishes", "Cauliflower Pepper Fry", 13.99),
    _m("Dry Dishes", "Chicken Podimas", 14.99),
    _m("Dry Dishes", "Mutton Pepper Fry", 16.99),
    _m("Dry Dishes", "Prawn Roast", 15.99),
    _m("Dry Dishes", "Nattukozhi Roast", 17.99),
    # Rice & Breads
    _m("Rice & Breads", "Plain Rice", 4.99),
    _m("Rice & Breads", "Curd Rice", 8.99),
    _m("Rice & Breads", "Parotta", 4.99, price_text="$4.99+"),
    _m("Rice & Breads", "Kal Dosai", price_text="$4.99-$9.99"),
    # Soups
    _m("Soups", "Rasam", 3.99),
    _m("Soups", "Mutton Bone Soup", 4.99),
    _m("Soups", "Crab Soup", 5.99),
    # Desserts
    _m("Desserts", "Gulab Jamoon", 4.99),
    _m("Desserts", "Pal Payasam", 4.99),
    _m("Desserts", "Pineapple Kesari", 4.99),
    _m("Desserts", "Elaneer Payasam", 6.99),
    _m("Desserts", "Ice Cream", 4.99),
    # Beverages
    _m("Beverages", "Tea", 2.49),
    _m("Beverages", "Madras Filter Coffee", 3.99),
    _m("Beverages", "Buttermilk", 3.99),
    _m("Beverages", "Mango Lassi", 5.49),
    _m("Beverages", "Rose Milk", 4.49),
    _m("Beverages", "Lemon Soda", 4.49),
    _m("Beverages", "Water", 0.99),
    # Sides
    _m("Sides", "Sambhar", 2.50, price_text="$2.50+"),
    _m("Sides", "Coconut Chutney", price_text="$2.49-$4.99"),
    _m("Sides", "Puli Kulambu", 4.99),
    _m("Sides", "Podi", 1.00),
    _m("Sides", "Ghee / Butter", 1.00),
    _m("Sides", "Pappad", 0.75),
]
