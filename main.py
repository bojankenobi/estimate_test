# -*- coding: utf-8 -*-
import streamlit as st
import sqlite3
import os
import math
import pandas as pd
import datetime
import io
import time
import logging

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Set level to INFO for production, DEBUG for development
logger = logging.getLogger(__name__)

# --- PRVA Streamlit komanda ---
st.set_page_config(page_title="Print Calculation", layout="wide")

# --- Konstante i podrazumevane vrednosti ---
PITCH = 3.175; GAP_MIN = 2.5; GAP_MAX = 4.0; Z_MIN = 70; Z_MAX = 140
TOTAL_CYLINDER_WIDTH = 200; WORKING_WIDTH = 190; WIDTH_GAP = 5
WIDTH_WASTE = 10; MAX_MATERIAL_WIDTH = 200
BASE_WASTE_LENGTH = 50.0; WASTE_LENGTH_PER_COLOR = 50.0
SETUP_TIME_PER_COLOR_OR_BASE = 30; CLEANUP_TIME_MIN = 30
MACHINE_SPEED_MIN = 10; MACHINE_SPEED_MAX = 120
GRAMS_INK_PER_M2 = 3.0; GRAMS_VARNISH_PER_M2 = 4.0

FALLBACK_INK_PRICE = 2350.0
FALLBACK_VARNISH_PRICE = 1800.0
FALLBACK_LABOR_PRICE = 3000.0
FALLBACK_TOOL_SEMI_PRICE = 6000.0
FALLBACK_TOOL_ROT_PRICE = 8000.0
FALLBACK_PLATE_PRICE = 2000.0
FALLBACK_SINGLE_PROFIT = 0.25
FALLBACK_MACHINE_SPEED = 30

FALLBACK_PROFITS = { 1000: 0.30, 10000: 0.25, 20000: 0.22, 50000: 0.20, 100000: 0.18 }
QUANTITIES_FOR_OFFER = list(FALLBACK_PROFITS.keys())

DB_FILE = "print_calculator.db"

# --- Database Connection Management with st.cache_resource ---
@st.cache_resource
def get_db_connection():
    """Uspostavlja konekciju sa SQLite bazom. Vraća konekciju."""
    logger.debug("Creating new SQLite connection")
    try:
        # check_same_thread=False is important for Streamlit
        conn = sqlite3.connect(DB_FILE, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to create database connection: {e}", exc_info=True)
        st.error(f"Failed to connect to database: {e}")
        return None

# --- Database Execution with Retry Logic ---
def execute_with_retry(query, params=(), retries=3, delay=1):
    """Izvršava SQL upit sa retry mehanizmom za slučaj zaključavanja baze."""
    conn = get_db_connection()
    if conn is None:
        logger.error("No database connection available for execute_with_retry")
        return False, None

    cursor = None  # Initialize cursor to None
    for attempt in range(retries):
        try:
            cursor = conn.cursor() # Assign cursor here
            logger.debug(f"Attempt {attempt+1}/{retries}: Executing query: {query[:100]}... with params: {params}")
            cursor.execute(query, params)

            is_select = query.strip().upper().startswith("SELECT")
            if is_select:
                result = cursor.fetchall()
                logger.debug(f"SELECT query successful, returning {len(result)} rows.")
                # No cursor.close() here, finally block handles it
                return True, result
            else:
                conn.commit()
                logger.debug("Non-SELECT query successful and committed.")
                # No cursor.close() here, finally block handles it
                return True, None

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < retries - 1:
                logger.warning(f"Database is locked on attempt {attempt + 1}, retrying in {delay}s...")
                if cursor: # Close cursor before sleeping if it was created
                    try:
                        cursor.close()
                        logger.debug("Cursor closed before retry due to lock.")
                    except sqlite3.Error as close_err:
                         logger.error(f"Error closing cursor before retry: {close_err}", exc_info=True)
                    cursor = None # Reset cursor
                time.sleep(delay)
                continue # Retry the loop
            else:
                logger.error(f"Database OperationalError on final attempt or non-lock error: {e}. Query: {query[:100]}...", exc_info=True)
                return False, None # Exit function after final retry or other OperationalError

        except sqlite3.Error as e:
            # Catch other SQLite errors
            logger.error(f"General Database Error: {e}. Query: {query[:100]}...", exc_info=True)
            try:
                 conn.rollback() # Rollback transaction on error for non-SELECT
                 logger.info("Transaction rolled back due to error.")
            except Exception as rb_err:
                 logger.error(f"Error during rollback: {rb_err}", exc_info=True)
            return False, None # Exit function

        finally:
            # Ensure cursor is closed if it was successfully created in this attempt
            if cursor:
                logger.debug(f"Closing cursor in finally block (Attempt {attempt+1})")
                try:
                    cursor.close()
                except sqlite3.Error as final_close_err:
                     logger.error(f"Error closing cursor in finally block: {final_close_err}", exc_info=True)
                # No need to set cursor=None here as it's the end of the try/except/finally for this attempt

    # If loop completes without returning True (all retries failed)
    logger.error(f"Query failed after {retries} retries, likely due to persistent locking. Query: {query[:100]}...")
    return False, None


@st.cache_data
def init_db():
    """Inicijalizuje bazu podataka. Vraća True ako uspe, False ako ne."""
    logger.info("Initializing database schema and default data...")
    overall_success = True
    try:
        # Create Tables
        table_queries = [
            """CREATE TABLE IF NOT EXISTS materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                price_per_m2 REAL NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY NOT NULL,
                value REAL NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS calculations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                client_name TEXT,
                product_name TEXT,
                template_width REAL,
                template_height REAL,
                quantity INTEGER,
                num_colors INTEGER,
                is_blank BOOLEAN,
                is_uv_varnish BOOLEAN,
                material_name TEXT,
                tool_type TEXT,       -- Store the descriptive string here
                machine_speed REAL,
                profit_coefficient REAL, -- The single one used for this specific calculation
                calculated_total_price REAL,
                calculated_price_per_piece REAL
                -- Add other fields if needed from calculation_data_for_db, e.g., specific costs
            )"""
        ]
        for query in table_queries:
            success, _ = execute_with_retry(query)
            if not success:
                logger.error(f"Failed to execute query during DB init (CREATE TABLE): {query}")
                overall_success = False
                # return False # Exit early if table creation fails

        # Check and Insert Default Materials
        success, result = execute_with_retry("SELECT COUNT(*) FROM materials")
        if success and result and result[0][0] == 0:
            logger.info("Materials table is empty, inserting default materials.")
            default_materials = {"Paper (chrome)": 39.95, "Plastic (PPW)": 54.05, "Thermal Paper": 49.35}
            for name, price in default_materials.items():
                mat_success, _ = execute_with_retry(
                    "INSERT INTO materials (name, price_per_m2) VALUES (?, ?)",
                    (name, price)
                )
                if not mat_success:
                    logger.error(f"Failed to insert default material: {name}")
                    overall_success = False # Mark failure but continue trying others
        elif not success:
             logger.error("Failed to count materials during DB init.")
             overall_success = False

        # Insert or Ignore Default Settings
        logger.info("Inserting or ignoring default settings.")
        default_settings = {
            "ink_price_per_kg": FALLBACK_INK_PRICE,
            "varnish_price_per_kg": FALLBACK_VARNISH_PRICE,
            "machine_labor_price_per_hour": FALLBACK_LABOR_PRICE,
            "tool_price_semirotary": FALLBACK_TOOL_SEMI_PRICE,
            "tool_price_rotary": FALLBACK_TOOL_ROT_PRICE,
            "plate_price_per_color": FALLBACK_PLATE_PRICE,
            "machine_speed_default": FALLBACK_MACHINE_SPEED,
            "single_calc_profit_coefficient": FALLBACK_SINGLE_PROFIT
        }
        for qty, coeff in FALLBACK_PROFITS.items():
            default_settings[f"profit_coeff_{qty}"] = coeff

        for key, value in default_settings.items():
            set_success, _ = execute_with_retry(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, value)
            )
            if not set_success:
                logger.warning(f"Failed to insert/ignore setting: {key}. Might already exist or DB issue.")
                # Depending on severity, you might set overall_success = False here
        logger.info("Finished inserting/ignoring default settings.")

    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {e}", exc_info=True)
        overall_success = False

    if overall_success:
        logger.info("Database initialization completed successfully.")
    else:
        logger.error("Database initialization failed or encountered errors.")

    return overall_success


# Using _conn argument pattern for potential cache invalidation on change
@st.cache_data(ttl=300) # Cache for 5 minutes
def load_materials_from_db(_conn):
    """Učitava materijale. Vraća rečnik ili None ako ne uspe."""
    logger.debug("Loading materials from database")
    success, result = execute_with_retry("SELECT name, price_per_m2 FROM materials ORDER BY name")
    if success:
        if result:
            logger.debug(f"Successfully loaded {len(result)} materials.")
            return {row['name']: row['price_per_m2'] for row in result}
        else:
            logger.info("Materials table is empty.")
            return {} # Return empty dict if table exists but is empty
    logger.error("Failed to load materials from database")
    return None # Indicate failure to load

@st.cache_data(ttl=300) # Cache for 5 minutes
def load_settings_from_db(_conn):
    """Učitava podešavanja. Vraća rečnik ili None ako ne uspe."""
    logger.debug("Loading settings from database")
    success, result = execute_with_retry("SELECT key, value FROM settings")
    if success:
        if result:
            logger.debug(f"Successfully loaded {len(result)} settings.")
            return {row['key']: row['value'] for row in result}
        else:
            logger.info("Settings table is empty.")
            return {} # Return empty dict if table exists but is empty
    logger.error("Failed to load settings from database")
    return None # Indicate failure to load

def update_material_price_in_db(name, price):
    """Ažurira cenu materijala i čisti relevantne keševe. Vraća True/False."""
    logger.debug(f"Attempting to update material price for '{name}' to {price}")
    success, _ = execute_with_retry(
        "UPDATE materials SET price_per_m2 = ? WHERE name = ?",
        (price, name)
    )
    if success:
        logger.info(f"Successfully updated material price for '{name}'. Clearing cache.")
        conn = get_db_connection()
        if conn:
            load_materials_from_db.clear() # Clear materials cache
    else:
         logger.error(f"Failed to update material price for '{name}'.")
    return success

def update_setting_in_db(key, value):
    """Ažurira podešavanje i čisti relevantne keševe. Vraća True/False."""
    logger.debug(f"Attempting to update setting '{key}' to {value}")
    success, _ = execute_with_retry(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        (key, value)
    )
    if success:
        logger.info(f"Successfully updated setting '{key}'. Clearing cache.")
        conn = get_db_connection()
        if conn:
             load_settings_from_db.clear() # Clear settings cache
    else:
        logger.error(f"Failed to update setting '{key}'.")
    return success

def save_calculation_to_db(calc_data):
    """Čuva podatke kalkulacije u bazu. Vraća True/False."""
    logger.debug("Attempting to save calculation to database")
    # Ensure we have data to save
    if not calc_data:
        logger.error("No calculation data provided to save.")
        return False

    sql = """INSERT INTO calculations (
        client_name, product_name, template_width, template_height, quantity,
        num_colors, is_blank, is_uv_varnish, material_name, tool_type,
        machine_speed, profit_coefficient, calculated_total_price, calculated_price_per_piece
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    # Extract data safely using .get() with defaults
    params = (
        calc_data.get('client_name'),
        calc_data.get('product_name'),
        calc_data.get('template_width_W_input'),
        calc_data.get('template_height_H_input'),
        calc_data.get('quantity_input'),
        calc_data.get('valid_num_colors_for_calc'),
        calc_data.get('is_blank'),
        calc_data.get('is_uv_varnish_input'),
        calc_data.get('selected_material'),
        calc_data.get('tool_info_string_final'), # Store the final descriptive tool string
        calc_data.get('machine_speed_m_min'),
        calc_data.get('profit_coefficient_used'), # The specific one used for this calc
        calc_data.get('total_selling_price_rsd'),
        calc_data.get('selling_price_per_piece_rsd')
    )
    logger.debug(f"Saving calculation with params: {params}")
    success, _ = execute_with_retry(sql, params)
    if success:
        logger.info("Calculation saved to database successfully.")
    else:
        logger.error("Failed to save calculation to database.")
        st.error("Error saving calculation to database!", icon="💾") # Keep UI feedback
    return success

# --- Database Functions (New for Search/Load) ---
# Consider caching search results for a short time if needed
# @st.cache_data(ttl=60)
def search_calculations(client_query=None, product_query=None, limit=50):
    """Pretražuje istoriju kalkulacija po klijentu i/ili proizvodu."""
    logger.info(f"Searching calculations with client='{client_query}', product='{product_query}'")
    base_query = """
        SELECT
            id,
            strftime('%Y-%m-%d %H:%M', timestamp) as Timestamp,
            client_name as Client,
            product_name as Product,
            quantity as Qty,
            calculated_total_price as TotalPrice
        FROM calculations
        WHERE 1=1
    """
    params = []
    # Append conditions and params only if query strings are provided and not empty
    if client_query and client_query.strip():
        base_query += " AND client_name LIKE ?"
        params.append(f"%{client_query.strip()}%")
    if product_query and product_query.strip():
        base_query += " AND product_name LIKE ?"
        params.append(f"%{product_query.strip()}%")

    base_query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    success, results = execute_with_retry(base_query, tuple(params))

    if success:
        if results:
            logger.info(f"Found {len(results)} matching calculations.")
            # Convert list of Row objects to list of dicts for DataFrame
            search_data = [dict(row) for row in results]
            return pd.DataFrame(search_data)
        else:
            logger.info("No matching calculations found.")
            return pd.DataFrame() # Return empty DataFrame if no results
    else:
        logger.error("Database error during calculation search.")
        st.error("Error searching calculation history.", icon="💾")
        return None # Indicate error

def load_calculation_details(calc_id):
    """Učitava sve relevantne detalje za dati ID kalkulacije."""
    if not isinstance(calc_id, int) or calc_id <= 0:
         logger.error(f"Invalid calculation ID provided for loading: {calc_id}")
         st.error(f"Invalid ID format: {calc_id}", icon="🔢")
         return None

    logger.info(f"Loading details for calculation ID: {calc_id}")
    query = "SELECT * FROM calculations WHERE id = ?"
    success, result = execute_with_retry(query, (calc_id,))

    if success:
        if result and len(result) == 1:
            logger.info(f"Successfully loaded details for calculation ID: {calc_id}")
            return dict(result[0]) # Returns row as a dictionary
        elif result and len(result) > 1:
             # This shouldn't happen if ID is primary key, but handle defensively
             logger.warning(f"Found multiple calculations for ID {calc_id}. Returning first one.")
             return dict(result[0])
        else:
            logger.warning(f"No calculation found with ID: {calc_id}")
            st.warning(f"No calculation found with ID: {calc_id}", icon="🤷")
            return None
    else:
        logger.error(f"Database error loading details for calculation ID: {calc_id}")
        st.error(f"Error loading calculation details for ID {calc_id}.", icon="💾")
        return None

# --- Calculation Functions ---
def find_cylinder_specifications(template_width_W):
    valid_solutions = []
    message = ""
    if template_width_W <= 0:
        return None, [], "Error: Template width must be > 0."
    # Loop through possible teeth counts
    for z in range(Z_MIN, Z_MAX + 1):
        circumference_C = z * PITCH
        # Check if template width + minimum gap can fit at all
        if (template_width_W + GAP_MIN) <= 1e-9: # Avoid division by zero or near-zero
            continue # Should not happen with W > 0 check, but good practice
        # Max possible number of templates around circumference
        n_max_possible = math.floor(circumference_C / (template_width_W + GAP_MIN))
        # Check potential numbers of templates (n) from 1 up to max possible
        for n in range(1, n_max_possible + 1):
            if n == 0: continue # Should not be reached due to range(1,...)

            # Calculate the resulting gap if n templates are used
            gap_G_circumference = (circumference_C / n) - template_width_W

            # Check if the calculated gap is within the allowed range (with tolerance)
            tolerance = 1e-9
            if (GAP_MIN - tolerance) <= gap_G_circumference <= (GAP_MAX + tolerance):
                valid_solutions.append({
                    "number_of_teeth_Z": z,
                    "circumference_mm": circumference_C,
                    "templates_N_circumference": n, # 'n' templates fit circumferentially
                    "gap_G_circumference_mm": gap_G_circumference
                })

    if not valid_solutions:
        message = f"No cylinder found ({Z_MIN}-{Z_MAX} teeth) for W={template_width_W:.3f}mm with G={GAP_MIN:.1f}-{GAP_MAX:.1f}mm."
        return None, [], message

    # Sort solutions: prioritize fewer teeth (smaller cylinder), then more templates (better utilization)
    valid_solutions.sort(key=lambda x: (x["number_of_teeth_Z"], -x["templates_N_circumference"]))

    # Return the best solution (first after sorting) and all valid solutions
    return valid_solutions[0], valid_solutions, "Circumference calculation OK."


def calculate_number_across_width(template_height_H, working_width, width_gap):
    """ Calculates how many templates (y) fit across the working width """
    if template_height_H <= 0:
        logger.warning("Template height H <= 0, returning 0 for number across width.")
        return 0
    if template_height_H > working_width:
        logger.debug(f"Template height {template_height_H} > working width {working_width}, only 0 fits.")
        return 0 # Template itself is wider than the working area

    # If one fits, but two (with gap) don't, then it's 1
    if template_height_H <= working_width and (template_height_H * 2 + width_gap) > working_width:
         logger.debug(f"One template fits ({template_height_H} <= {working_width}), but two don't ({(template_height_H * 2 + width_gap)} > {working_width}). Returning 1.")
         return 1

    # General case: how many (Height + Gap) units fit into (WorkingWidth + Gap)?
    # Adding the gap to working width accounts for the fact that the last template doesn't need a gap *after* it within the working width.
    denominator = template_height_H + width_gap
    if denominator <= 1e-9: # Avoid division by zero
        logger.warning("Template height + width gap is near zero, returning 0.")
        return 0

    num_across = int(math.floor((working_width + width_gap) / denominator))
    logger.debug(f"Calculated number across width (y) = floor(({working_width} + {width_gap}) / ({template_height_H} + {width_gap})) = {num_across}")
    return num_across


def calculate_material_width(number_across_width_y, template_height_H, width_gap, width_waste):
    """ Calculates the total required material width based on templates, gaps, and waste """
    if number_across_width_y <= 0:
        logger.warning("Number across width is <= 0, returning 0 for material width.")
        return 0
    # Total width taken by templates
    total_template_width = number_across_width_y * template_height_H
    # Total width taken by gaps between templates (n templates have n-1 gaps)
    total_gap_width = max(0, number_across_width_y - 1) * width_gap
    # Total required material width is templates + gaps + side waste
    required_width = total_template_width + total_gap_width + width_waste
    logger.debug(f"Calculated material width: ({number_across_width_y} * {template_height_H}) + ({max(0, number_across_width_y - 1)} * {width_gap}) + {width_waste} = {required_width}")
    return required_width


def format_time(total_minutes):
    """ Formats total minutes into hours and minutes string """
    if total_minutes is None or total_minutes < 0:
        return "N/A"
    total_minutes = round(total_minutes) # Round to nearest minute
    if total_minutes == 0:
        return "0 min"
    if total_minutes < 60:
        return f"{int(total_minutes)} min" # Ensure integer output
    else:
        hours, minutes = divmod(total_minutes, 60)
        hours = int(hours)
        minutes = int(minutes)
        if minutes == 0:
            return f"{hours} h"
        else:
            return f"{hours} h {minutes} min"

# --- Core Calculation Logic ---
def run_single_calculation(quantity: int, template_width_W: float, template_height_H: float,
                           best_circumference_solution: dict, number_across_width_y: int, is_blank: bool,
                           num_colors: int, is_uv_varnish: bool, price_per_m2: float, machine_speed_m_min: float,
                           selected_tool_key: str, existing_tool_info: str, profit_coefficient: float,
                           ink_price_kg: float, varnish_price_kg: float, plate_price_color: float,
                           labor_price_hour: float, tool_price_semi: float, tool_price_rot: float) -> dict:
    """ Performs a complete cost calculation for a single quantity """
    results = {}
    logger.info(f"--- Running calculation for Qty: {quantity:,} ---")
    logger.debug(f"Inputs: W={template_width_W}, H={template_height_H}, Y={number_across_width_y}, "
                 f"Blank={is_blank}, Colors={num_colors}, Varnish={is_uv_varnish}, MatPrice={price_per_m2:.2f}, "
                 f"Speed={machine_speed_m_min:.0f}, Tool={selected_tool_key}, ToolInfo='{existing_tool_info}', "
                 f"Profit={profit_coefficient:.3f}, InkP={ink_price_kg:.2f}, VarnP={varnish_price_kg:.2f}, PlateP={plate_price_color:.2f}, "
                 f"LaborP={labor_price_hour:.2f}, ToolSemiP={tool_price_semi:.2f}, ToolRotP={tool_price_rot:.2f}")
    logger.debug(f"Cylinder Spec: {best_circumference_solution}")


    # --- Initial Checks ---
    if not best_circumference_solution or number_across_width_y <= 0:
        error_msg = "Invalid configuration: "
        if not best_circumference_solution: error_msg += "No valid cylinder solution found. "
        if number_across_width_y <= 0: error_msg += "Number across width (y) must be > 0."
        logger.error(error_msg)
        results['error'] = error_msg
        results['total_selling_price_rsd'] = 0.0
        results['selling_price_per_piece_rsd'] = 0.0
        return results
    if price_per_m2 < 0: logger.warning("Material price is negative.")
    if machine_speed_m_min <= 0: logger.warning("Machine speed is zero or negative.")
    if profit_coefficient is None or profit_coefficient < 0:
        logger.error(f"Invalid profit coefficient: {profit_coefficient}. Using 0.")
        profit_coefficient = 0.0


    # --- Calculations ---
    # 1. Material Width
    gap_G_circumference_mm = best_circumference_solution.get('gap_G_circumference_mm', 0.0) # Get gap from solution
    results['gap_G_circumference_mm'] = gap_G_circumference_mm # Store it in results
    required_material_width_mm = calculate_material_width(number_across_width_y, template_height_H, WIDTH_GAP, WIDTH_WASTE)
    results['required_material_width_mm'] = required_material_width_mm
    results['material_width_exceeded'] = required_material_width_mm > MAX_MATERIAL_WIDTH

    # 2. Material Consumption (Length & Area)
    total_production_length_m = 0.0
    total_production_area_m2 = 0.0
    if number_across_width_y > 0 and quantity > 0:
        # Length of one segment (template + gap) along the circumference
        segment_length_mm = template_width_W + gap_G_circumference_mm
        # Total length needed = (Total Quantity / Templates per Segment Widthwise) * Segment Length
        total_production_length_m = (quantity / number_across_width_y) * (segment_length_mm / 1000.0) # Convert mm to m
    if required_material_width_mm > 0:
        # Total area = Total Length (m) * Material Width (m)
        total_production_area_m2 = total_production_length_m * (required_material_width_mm / 1000.0) # Convert mm to m

    results['total_production_length_m'] = total_production_length_m
    results['total_production_area_m2'] = total_production_area_m2

    # 3. Waste Material (Setup)
    # Use 1 "color" for setup time/waste if blank, otherwise actual color count
    num_setups_for_waste_time = 1 if is_blank else (num_colors if num_colors > 0 else 1) # Ensure at least 1 setup
    # Varnish doesn't add extra setup waste length/time here, only consumption/cost later
    waste_length_m = BASE_WASTE_LENGTH + (0 if is_blank else (num_colors * WASTE_LENGTH_PER_COLOR))
    waste_area_m2 = waste_length_m * (required_material_width_mm / 1000.0) if required_material_width_mm > 0 else 0.0
    results['waste_length_m'] = waste_length_m
    results['waste_area_m2'] = waste_area_m2

    # 4. Total Material (Production + Waste)
    total_final_length_m = total_production_length_m + waste_length_m
    total_final_area_m2 = total_production_area_m2 + waste_area_m2
    results['total_final_length_m'] = total_final_length_m
    results['total_final_area_m2'] = total_final_area_m2

    # 5. Time Calculation
    setup_time_min = num_setups_for_waste_time * SETUP_TIME_PER_COLOR_OR_BASE
    production_time_min = (total_production_length_m / machine_speed_m_min) if machine_speed_m_min > 0 else 0.0
    cleanup_time_min = CLEANUP_TIME_MIN # Constant cleanup time
    total_time_min = setup_time_min + production_time_min + cleanup_time_min
    results['setup_time_min'] = setup_time_min
    results['production_time_min'] = production_time_min
    results['cleanup_time_min'] = cleanup_time_min
    results['total_time_min'] = total_time_min

    # 6. Cost Calculation
    # Ink & Varnish Cost
    ink_cost_rsd = 0.0
    ink_consumption_kg = 0.0
    varnish_cost_rsd = 0.0
    varnish_consumption_kg = 0.0
    # Ensure prices are non-negative
    safe_ink_price_kg = max(0, ink_price_kg)
    safe_varnish_price_kg = max(0, varnish_price_kg)
    safe_plate_price_color = max(0, plate_price_color)
    safe_price_per_m2 = max(0, price_per_m2)
    safe_labor_price_hour = max(0, labor_price_hour)
    safe_tool_price_semi = max(0, tool_price_semi)
    safe_tool_price_rot = max(0, tool_price_rot)


    if not is_blank and num_colors > 0 and total_production_area_m2 > 0:
        # Ink Consumption (kg) = Area (m²) * Num Colors * Grams/m²/Color / 1000 (g to kg)
        ink_consumption_kg = (total_production_area_m2 * num_colors * GRAMS_INK_PER_M2) / 1000.0
        ink_cost_rsd = ink_consumption_kg * safe_ink_price_kg
    if is_uv_varnish and total_production_area_m2 > 0:
        # Varnish Consumption (kg) = Area (m²) * Grams/m² / 1000 (g to kg)
        varnish_consumption_kg = (total_production_area_m2 * GRAMS_VARNISH_PER_M2) / 1000.0
        varnish_cost_rsd = varnish_consumption_kg * safe_varnish_price_kg
    total_ink_varnish_cost_rsd = ink_cost_rsd + varnish_cost_rsd
    results['ink_cost_rsd'] = ink_cost_rsd
    results['varnish_cost_rsd'] = varnish_cost_rsd
    results['total_ink_varnish_cost_rsd'] = total_ink_varnish_cost_rsd # Store combined cost too

    # Plate Cost
    total_plate_cost_rsd = (num_colors * safe_plate_price_color) if not is_blank and num_colors > 0 else 0.0
    results['plate_cost_rsd'] = total_plate_cost_rsd

    # Material Cost
    total_material_cost_rsd = total_final_area_m2 * safe_price_per_m2 if total_final_area_m2 > 0 else 0.0
    results['material_cost_rsd'] = total_material_cost_rsd

    # Machine Labor Cost
    total_machine_labor_cost_rsd = (total_time_min / 60.0) * safe_labor_price_hour if total_time_min > 0 else 0.0
    results['labor_cost_rsd'] = total_machine_labor_cost_rsd

    # Tool Cost (One-time, added to this specific calculation)
    total_tool_cost_rsd = 0.0
    # Determine tool info string based on selection AND input
    tool_info_string_final = selected_tool_key # Default
    if selected_tool_key == "None":
        # Only use the text input if it's not empty
        tool_info_string_final = f"Existing: {existing_tool_info.strip()}" if existing_tool_info and existing_tool_info.strip() else "None (No Tool Cost)"
        total_tool_cost_rsd = 0.0 # Explicitly zero cost for 'None' or 'Existing'
        logger.debug("Tool type is None/Existing, tool cost set to 0.")
    elif selected_tool_key == "Semirotary":
        total_tool_cost_rsd = safe_tool_price_semi
        tool_info_string_final = f"Semirotary ({total_tool_cost_rsd:,.0f} RSD)"
        logger.debug(f"Tool type is Semirotary, cost: {total_tool_cost_rsd}")
    elif selected_tool_key == "Rotary":
        total_tool_cost_rsd = safe_tool_price_rot
        tool_info_string_final = f"Rotary ({total_tool_cost_rsd:,.0f} RSD)"
        logger.debug(f"Tool type is Rotary, cost: {total_tool_cost_rsd}")
    else:
        # Should not happen with radio buttons, but handle defensively
         logger.warning(f"Unknown tool key selected: {selected_tool_key}, setting tool cost to 0.")
         tool_info_string_final = f"Unknown ({selected_tool_key})"
         total_tool_cost_rsd = 0.0

    results['tool_cost_rsd'] = total_tool_cost_rsd
    results['tool_info_string_final'] = tool_info_string_final # Store the final descriptive string

    # Total Production Cost
    total_production_cost_rsd = (total_ink_varnish_cost_rsd +
                                total_plate_cost_rsd +
                                total_material_cost_rsd +
                                total_machine_labor_cost_rsd +
                                total_tool_cost_rsd)
    results['total_production_cost_rsd'] = total_production_cost_rsd

    # 7. Profit Calculation
    # Profit is calculated as a percentage of the MATERIAL cost only
    profit_rsd = total_material_cost_rsd * profit_coefficient if total_material_cost_rsd > 0 else 0.0
    #profit_rsd = total_production_cost_rsd * profit_coefficient # Alternative: Profit on total production cost
    results['profit_rsd'] = profit_rsd
    results['profit_coefficient_used'] = profit_coefficient # Store which coefficient was used

    # 8. Final Selling Price
    total_selling_price_rsd = total_production_cost_rsd + profit_rsd
    selling_price_per_piece_rsd = (total_selling_price_rsd / quantity) if quantity > 0 else 0.0
    results['total_selling_price_rsd'] = total_selling_price_rsd
    results['selling_price_per_piece_rsd'] = selling_price_per_piece_rsd

    # --- Logging Summary ---
    logger.info(f"Calculation Summary (Qty: {quantity:,}):")
    logger.info(f"  Material: Width={required_material_width_mm:.2f}mm, Exceeded={results['material_width_exceeded']}, "
                f"Total Area={total_final_area_m2:.2f}m²")
    logger.info(f"  Time: Setup={format_time(setup_time_min)}, Prod={format_time(production_time_min)}, "
                f"Clean={format_time(cleanup_time_min)}, TOTAL={format_time(total_time_min)}")
    logger.info(f"  Costs (RSD): Ink/Varnish={total_ink_varnish_cost_rsd:,.2f}, Plates={total_plate_cost_rsd:,.2f}, "
                f"Material={total_material_cost_rsd:,.2f}, Labor={total_machine_labor_cost_rsd:,.2f}, Tool={total_tool_cost_rsd:,.2f}")
    logger.info(f"  Total Prod Cost: {total_production_cost_rsd:,.2f} RSD")
    logger.info(f"  Profit: {profit_rsd:,.2f} RSD (Coeff: {profit_coefficient:.3f} on Material Cost)")
    logger.info(f"  FINAL PRICE: Total={total_selling_price_rsd:,.2f} RSD, Per Piece={selling_price_per_piece_rsd:.4f} RSD")
    logger.info(f"--- Calculation End for Qty: {quantity:,} ---")

    return results

# --- PDF Generation Functions ---
def create_pdf(data):
    """Generates the detailed Calculation PDF report."""
    buffer = io.BytesIO()
    styles = getSampleStyleSheet() # Get standard styles

    # Custom styles based on ReportLab defaults
    styleH1 = ParagraphStyle(name='CustomH1', parent=styles['h1'], alignment=TA_CENTER, spaceAfter=6*mm)
    styleH2 = ParagraphStyle(name='CustomH2', parent=styles['h2'], spaceBefore=6*mm, spaceAfter=3*mm)
    styleH3 = ParagraphStyle(name='CustomH3', parent=styles['h3'], spaceBefore=4*mm, spaceAfter=2*mm, fontSize=10)
    styleN = ParagraphStyle(name='CustomNormal', parent=styles['Normal'], leading=14) # Normal text with slightly more leading
    styleSmallText = ParagraphStyle(name='SmallText', parent=styleN, fontSize=8, leading=10) # Smaller text style

    # Helper for creating bold paragraphs with a specific style
    def bold_paragraph(text, style=styleN):
        return Paragraph(f"<b>{str(text)}</b>", style) # Ensure text is string

    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=15*mm, bottomMargin=15*mm,
                            title=f"Calculation_{data.get('product_name', 'product')}",
                            author="Print Calculator")
    story = []

    # --- Header ---
    story.append(Paragraph("Print Calculation Report", styleH1))
    story.append(Spacer(1, 6*mm))
    header_data = [
        [bold_paragraph("Client:", style=styleN), Paragraph(f"{data.get('client_name', 'N/A')}", styleN)],
        [bold_paragraph("Product/Label:", style=styleN), Paragraph(f"{data.get('product_name', 'N/A')}", styleN)],
        [bold_paragraph("Date Generated:", style=styleN), Paragraph(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styleN)]
    ]
    header_table = Table(header_data, colWidths=[40*mm, 140*mm])
    header_table.setStyle(TableStyle([
       ('VALIGN', (0, 0), (-1, -1), 'TOP'),
       ('LEFTPADDING', (0, 0), (-1, -1), 0),
       ('BOTTOMPADDING', (0, 0), (-1, -1), 1*mm),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6*mm))

    # --- Input Parameters ---
    story.append(Paragraph("Input Parameters Summary", styleH2))
    num_colors_display = 'Blank' if data.get('is_blank') else f"{data.get('valid_num_colors_for_calc', 'N/A')}"
    params_data = [
        [bold_paragraph('Parameter', style=styleN), bold_paragraph('Value', style=styleN)],
        ['Template Width (W)', f"{data.get('template_width_W_input', 'N/A'):.3f} mm"],
        ['Template Height (H)', f"{data.get('template_height_H_input', 'N/A'):.3f} mm"],
        ['Desired Quantity', f"{data.get('quantity_input', 'N/A'):,}"],
        ['Colors', num_colors_display],
        ['UV Varnish', 'Yes' if data.get('is_uv_varnish_input') else 'No'],
        ['Material', f"{data.get('selected_material', 'N/A')} ({data.get('price_per_m2', 0.0):.2f} RSD/m²)"],
        ['Tool', f"{data.get('tool_info_string_final', 'N/A')}"], # Use the final descriptive string
        ['Machine Speed', f"{data.get('machine_speed_m_min', 'N/A'):.0f} m/min"],
        # Ensure profit coefficient is formatted correctly, handle None
        ['Profit Coefficient (Used)', f"{data.get('profit_coefficient_used', 0.0):.3f} (on material cost)" if data.get('profit_coefficient_used') is not None else "N/A"]
    ]
    # Convert simple strings in params_data to Paragraph objects for consistent styling
    params_data_p = [[item if isinstance(item, Paragraph) else Paragraph(str(item), styleN) for item in row] for row in params_data]
    # Apply bold style to the first column header specifically
    params_data_p[0][0] = bold_paragraph('Parameter', style=styleN)
    params_data_p[0][1] = bold_paragraph('Value', style=styleN)

    params_table = Table(params_data_p, colWidths=[60*mm, 120*mm])
    params_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), # Handled by bold_paragraph
        # ('FONTSIZE', (0, 0), (-1, 0), 10), # Already default size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 3*mm),
        ('TOPPADDING', (0, 0), (-1, 0), 2*mm),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1,-1), 2*mm),
        ('RIGHTPADDING', (0, 0), (-1,-1), 2*mm),
    ]))
    story.append(params_table)
    story.append(Spacer(1, 6*mm))

    # --- Calculation Results ---
    story.append(Paragraph("Calculation Results", styleH2))

    # 1. Cylinder and Template Configuration
    story.append(Paragraph("1. Cylinder and Template Configuration", styleH3))
    bc_sol = data.get('best_circumference_solution', {})
    config_data = [
        [bold_paragraph('Item'), bold_paragraph('Value')], # Header row
        ['Number of Teeth (Z)', f"{bc_sol.get('number_of_teeth_Z', 'N/A')}"],
        ['Cylinder Circumference', f"{bc_sol.get('circumference_mm', 0.0):.3f} mm"],
        ['Circumference Gap (G)', f"{data.get('gap_G_circumference_mm', 0.0):.3f} mm"],
        ['Templates Circumference (x)', f"{bc_sol.get('templates_N_circumference', 'N/A')}"], # Use correct key from find_cylinder
        ['Templates Width (y)', f"{data.get('number_across_width_y', 'N/A')}"],
        ['Format (y × x)', f"{data.get('number_across_width_y', 'N/A')} × {bc_sol.get('templates_N_circumference', 'N/A')}"]
    ]
     # Convert simple strings to Paragraph objects
    config_data_p = [[item if isinstance(item, Paragraph) else Paragraph(str(item), styleN) for item in row] for row in config_data]
    config_data_p[0][0] = bold_paragraph('Item', style=styleN)
    config_data_p[0][1] = bold_paragraph('Value', style=styleN)

    config_table = Table(config_data_p, colWidths=[60*mm, 120*mm])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        # ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), # Handled by bold_paragraph
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3*mm),
        ('TOPPADDING', (0, 0), (-1, -1), 2*mm),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1,-1), 2*mm),
    ]))
    story.append(config_table)
    story.append(Spacer(1, 4*mm))

    # 2. Material Width
    story.append(Paragraph("2. Material Width", styleH3))
    mat_width_status = f"OK (≤ {MAX_MATERIAL_WIDTH} mm)" if not data.get('material_width_exceeded') else f"⚠️ EXCEEDED! (> {MAX_MATERIAL_WIDTH} mm)"
    # Use HTML color names recognized by ReportLab Paragraphs
    color_name = "green" if not data.get('material_width_exceeded') else "red"
    story.append(Paragraph(f"Required Material Width: {data.get('required_material_width_mm', 0.0):.2f} mm (<font color='{color_name}'>{mat_width_status}</font>)", styleN))
    story.append(Spacer(1, 4*mm))

    # 3. Material Consumption
    story.append(Paragraph("3. Material Consumption", styleH3))
    # Use the defined styleSmallText for headers
    consumption_data_styled = [
        [bold_paragraph('Category', styleSmallText), bold_paragraph('Length (m)', styleSmallText), bold_paragraph('Area (m²)', styleSmallText)],
        [Paragraph('Production', styleN), Paragraph(f"{data.get('total_production_length_m', 0.0):,.2f}", styleN), Paragraph(f"{data.get('total_production_area_m2', 0.0):,.2f}", styleN)],
        [Paragraph('Waste (Setup)', styleN), Paragraph(f"{data.get('waste_length_m', 0.0):,.2f}", styleN), Paragraph(f"{data.get('waste_area_m2', 0.0):,.2f}", styleN)],
        [bold_paragraph('TOTAL', styleN), bold_paragraph(f"{data.get('total_final_length_m', 0.0):,.2f}", styleN), bold_paragraph(f"{data.get('total_final_area_m2', 0.0):,.2f}", styleN)]
    ]
    consumption_table = Table(consumption_data_styled, colWidths=[60*mm, 60*mm, 60*mm])
    consumption_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'), # Align numbers right (except header which is handled by Paragraph)
        ('ALIGN', (0, 0), (0, -1), 'LEFT'), # Align first column left
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        # ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), # Handled by bold_paragraph
        ('FONTNAME', (0, 3), (-1, 3), 'Helvetica-Bold'), # Handled by bold_paragraph
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3*mm),
        ('TOPPADDING', (0, 0), (-1, -1), 2*mm),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('LEFTPADDING', (0, 0), (0,-1), 2*mm),
    ]))
    story.append(consumption_table)
    story.append(Spacer(1, 4*mm))

    # 4. Estimated Production Time
    story.append(Paragraph("4. Estimated Production Time", styleH3))
    time_data_styled = [
        [bold_paragraph('Phase'), bold_paragraph('Duration')], # Header
        ['Setup Time', format_time(data.get('setup_time_min', 0.0))],
        ['Production Time', format_time(data.get('production_time_min', 0.0))],
        ['Cleanup Time', format_time(data.get('cleanup_time_min', 0.0))],
        [bold_paragraph('TOTAL Work Time'), bold_paragraph(format_time(data.get('total_time_min', 0.0)))]
    ]
     # Convert simple strings to Paragraph objects
    time_data_p = [[item if isinstance(item, Paragraph) else Paragraph(str(item), styleN) for item in row] for row in time_data_styled]
    time_data_p[0][0] = bold_paragraph('Phase', style=styleN)
    time_data_p[0][1] = bold_paragraph('Duration', style=styleN)
    time_data_p[4][0] = bold_paragraph('TOTAL Work Time', style=styleN)
    time_data_p[4][1] = bold_paragraph(format_time(data.get('total_time_min', 0.0)), style=styleN)

    time_table = Table(time_data_p, colWidths=[60*mm, 120*mm])
    time_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        # ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), # Handled by bold_paragraph
        # ('FONTNAME', (0, 4), (-1, 4), 'Helvetica-Bold'), # Handled by bold_paragraph
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3*mm),
        ('TOPPADDING', (0, 0), (-1, -1), 2*mm),
        ('LEFTPADDING', (0, 0), (-1,-1), 2*mm),
    ]))
    story.append(time_table)
    story.append(Spacer(1, 6*mm))

    # --- Cost Calculation ---
    story.append(Paragraph("Cost Calculation", styleH2))
    cost_data_styled = [
        [bold_paragraph('Cost Item'), bold_paragraph('Amount (RSD)')], # Header
        ['Ink', f"{data.get('ink_cost_rsd', 0.0):,.2f}"],
        ['Varnish', f"{data.get('varnish_cost_rsd', 0.0):,.2f}"],
        ['Plates', f"{data.get('plate_cost_rsd', 0.0):,.2f}"],
        ['Material', f"{data.get('material_cost_rsd', 0.0):,.2f}"],
        ['Tool', f"{data.get('tool_cost_rsd', 0.0):,.2f}"], # Tool cost from calculation
        ['Machine Labor', f"{data.get('labor_cost_rsd', 0.0):,.2f}"],
        [bold_paragraph('Total Production Cost'), bold_paragraph(f"{data.get('total_production_cost_rsd', 0.0):,.2f}")]
    ]
     # Convert simple strings to Paragraph objects
    cost_data_p = [[item if isinstance(item, Paragraph) else Paragraph(str(item), styleN) for item in row] for row in cost_data_styled]
    cost_data_p[0][0] = bold_paragraph('Cost Item', style=styleN)
    cost_data_p[0][1] = bold_paragraph('Amount (RSD)', style=styleN)
    cost_data_p[7][0] = bold_paragraph('Total Production Cost', style=styleN)
    cost_data_p[7][1] = bold_paragraph(f"{data.get('total_production_cost_rsd', 0.0):,.2f}", style=styleN)


    cost_table = Table(cost_data_p, colWidths=[100*mm, 80*mm])
    cost_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'), # Align amounts right
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        # ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), # Handled by bold_paragraph
        # ('FONTNAME', (0, 7), (-1, 7), 'Helvetica-Bold'), # Handled by bold_paragraph
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3*mm),
        ('TOPPADDING', (0, 0), (-1, -1), 2*mm),
        ('BACKGROUND', (0, 1), (-1, -2), colors.lightgreen), # Light green for items
        ('BACKGROUND', (0, 7), (-1, 7), colors.darkseagreen), # Darker green for total
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('LEFTPADDING', (0, 0), (0,-1), 2*mm),
        ('RIGHTPADDING', (1, 0), (-1,-1), 2*mm),
    ]))
    story.append(cost_table)
    story.append(Spacer(1, 6*mm))

    # --- Final Price Summary ---
    story.append(Paragraph("Final Price Summary", styleH2))
    profit_perc_str = f"({data.get('profit_coefficient_used', 0.0)*100:.1f}%)" if data.get('profit_coefficient_used') is not None else ""
    final_price_data_styled = [
        [bold_paragraph('Item'), bold_paragraph('Amount (RSD)')], # Header
        ['Total Production Cost', f"{data.get('total_production_cost_rsd', 0.0):,.2f}"],
        ['Profit', f"{data.get('profit_rsd', 0.0):,.2f} {profit_perc_str}"],
        [bold_paragraph('TOTAL PRICE (Selling)'), bold_paragraph(f"{data.get('total_selling_price_rsd', 0.0):,.2f}")],
        [bold_paragraph('Selling Price per Piece'), bold_paragraph(f"{data.get('selling_price_per_piece_rsd', 0.0):.4f}")]
    ]
     # Convert simple strings to Paragraph objects
    final_price_data_p = [[item if isinstance(item, Paragraph) else Paragraph(str(item), styleN) for item in row] for row in final_price_data_styled]
    final_price_data_p[0][0] = bold_paragraph('Item', style=styleN)
    final_price_data_p[0][1] = bold_paragraph('Amount (RSD)', style=styleN)
    final_price_data_p[3][0] = bold_paragraph('TOTAL PRICE (Selling)', style=styleN)
    final_price_data_p[3][1] = bold_paragraph(f"{data.get('total_selling_price_rsd', 0.0):,.2f}", style=styleN)
    final_price_data_p[4][0] = bold_paragraph('Selling Price per Piece', style=styleN)
    final_price_data_p[4][1] = bold_paragraph(f"{data.get('selling_price_per_piece_rsd', 0.0):.4f}", style=styleN)


    final_price_table = Table(final_price_data_p, colWidths=[100*mm, 80*mm])
    final_price_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'), # Align amounts right
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        # ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), # Handled by bold_paragraph
        # ('FONTNAME', (0, 3), (-1, 4), 'Helvetica-Bold'), # Handled by bold_paragraph
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3*mm),
        ('TOPPADDING', (0, 0), (-1, -1), 2*mm),
        ('BACKGROUND', (0, 1), (-1, 2), colors.antiquewhite), # Background for cost/profit
        ('BACKGROUND', (0, 3), (-1, 4), colors.lightcoral),   # Background for totals
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('LEFTPADDING', (0, 0), (0,-1), 2*mm),
        ('RIGHTPADDING', (1, 0), (-1,-1), 2*mm),
    ]))
    story.append(final_price_table)

    # --- Build PDF ---
    try:
        doc.build(story)
        buffer.seek(0)
        logger.info("Calculation PDF generated successfully.")
        return buffer
    except Exception as e:
        logger.error(f"Error building Calculation PDF: {e}", exc_info=True)
        st.error(f"Error building PDF: {e}")
        return None

def create_offer_pdf(data):
    """Generates the multi-quantity Offer PDF."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm,
                            title=f"Offer_{data.get('product_name', 'product')}",
                            author="Print Calculator")
    styles = getSampleStyleSheet()
    story = []

    # Custom Styles
    styleNormal = ParagraphStyle(name='OfferNormal', parent=styles['Normal'], leading=14)
    styleH1_Offer = ParagraphStyle(name='OfferTitle', parent=styles['h1'], alignment=TA_CENTER, spaceAfter=10*mm)
    styleH2_Offer = ParagraphStyle(name='OfferHeading', parent=styles['h2'], spaceBefore=6*mm, spaceAfter=4*mm, fontSize=12)
    styleItalic = ParagraphStyle(name='ItalicText', parent=styleNormal, fontName='Helvetica-Oblique', fontSize=9, alignment=TA_RIGHT)
    styleBold = ParagraphStyle(name='BoldText', parent=styleNormal, fontName='Helvetica-Bold')

    def bold_paragraph(text, style=styleNormal):
        return Paragraph(f"<b>{str(text)}</b>", style)

    # --- Offer Header ---
    story.append(Paragraph("PONUDA / OFFER", styleH1_Offer))
    offer_date = datetime.datetime.now().strftime('%d.%m.%Y')
    story.append(Paragraph(f"<b>Datum / Date:</b> {offer_date}", styleNormal))
    story.append(Paragraph(f"<b>Za / To:</b> {data.get('client_name', 'N/A')}", styleNormal))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(f"<b>Predmet / Subject:</b> Ponuda za izradu samolepljivih etiketa / Offer for self-adhesive label production", styleNormal))
    story.append(Paragraph(f"<b>Proizvod / Product:</b> {data.get('product_name', 'N/A')}", styleNormal))
    story.append(Spacer(1, 6*mm))

    # --- Specification ---
    story.append(Paragraph("Specifikacija / Specification:", styleH2_Offer))
    spec_list_data = data.get('specifications', {})
    # Ensure values are strings for Paragraph
    spec_table_data = [[bold_paragraph(k, style=styleNormal), Paragraph(str(v), styleNormal)] for k, v in spec_list_data.items()]
    spec_table = Table(spec_table_data, colWidths=[50*mm, 120*mm])
    spec_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey), # Lighter grid
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3*mm),
        ('TOPPADDING', (0, 0), (-1, -1), 3*mm),
        ('LEFTPADDING', (0, 0), (-1,-1), 2*mm),
    ]))
    story.append(spec_table)
    story.append(Spacer(1, 8*mm))

    # --- Prices ---
    story.append(Paragraph("Cene / Prices:", styleH2_Offer))
    offer_results = data.get('offer_results', [])
    if offer_results:
        # Create header row with bold Paragraphs using the bold style
        price_table_header = [
            bold_paragraph("Količina (kom)", styleBold),
            bold_paragraph("Cena/kom (RSD)", styleBold),
            bold_paragraph("Ukupno (RSD)", styleBold)
        ]
        price_table_data = [price_table_header]
        # Create data rows with formatted numbers as strings in Paragraphs
        for row in offer_results:
            price_table_data.append([
                Paragraph(f"{row.get('Količina (kom)', 0):,}", styleNormal),
                Paragraph(f"{row.get('Cena/kom (RSD)', 0.0):.4f}", styleNormal),
                Paragraph(f"{row.get('Ukupno (RSD)', 0.0):,.2f}", styleNormal)
            ])

        price_table = Table(price_table_data, colWidths=[50*mm, 55*mm, 65*mm])
        price_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'), # Center header text
            ('ALIGN', (0, 1), (0, -1), 'RIGHT'),  # Align quantity right
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'), # Align prices right
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            # ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), # Handled by bold_paragraph
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3*mm),
            ('TOPPADDING', (0, 0), (-1, -1), 3*mm),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('RIGHTPADDING', (0,0), (-1,-1), 2*mm),
        ]))
        story.append(price_table)
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph("<i>Napomena: U cene nije uključen PDV. Cena alata je uključena u ukupnu cenu (ako je primenjivo).</i>", styleItalic))
        story.append(Paragraph("<i>Note: VAT is not included. Tool cost is included in the total price (if applicable).</i>", styleItalic))
    else:
        story.append(Paragraph("Nema dostupnih cena za prikaz.", styleNormal))

    # --- Terms ---
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph("<b>Rok isporuke / Delivery Time:</b> Po dogovoru / As agreed", styleNormal))
    story.append(Paragraph("<b>Paritet / Incoterms:</b> FCO magacin Kupca / FCO Buyer's warehouse", styleNormal))
    story.append(Paragraph("<b>Validnost ponude / Offer Validity:</b> 15 dana / 15 days", styleNormal))

    # --- Build Offer PDF ---
    try:
        doc.build(story)
        buffer.seek(0)
        logger.info("Offer PDF generated successfully.")
        return buffer
    except Exception as e:
        logger.error(f"Error building Offer PDF: {e}", exc_info=True)
        st.error(f"Error building Offer PDF: {e}")
        return None


# --- Helper funkcija za sinhronizaciju UI -> State -> DB ---
def synced_number_input(label, state_key, db_key, min_val=0.0, max_val=None, step=1.0, format_str="%.2f", help_text=None, is_profit_coeff=False, disabled=False): # Added 'disabled' argument
    """
    Manages a number input, synchronizing Streamlit state and database.
    Uses a unique key for the widget itself to avoid conflicts.
    Handles potential loading errors by using fallbacks.
    Accepts and passes the 'disabled' argument to the underlying widget.
    """
    widget_key = f"{state_key}_input_widget_{label.replace(' ', '_').lower()}" # More unique key

    # Initialize state from DB settings if not already present
    if state_key not in st.session_state:
        # Determine fallback value based on context
        fallback_val = None
        if is_profit_coeff:
            try:
                 qty_suffix = int(db_key.split('_')[-1])
                 fallback_val = FALLBACK_PROFITS.get(qty_suffix, FALLBACK_SINGLE_PROFIT)
            except (IndexError, ValueError):
                 fallback_val = FALLBACK_SINGLE_PROFIT # Fallback for single coeff or parse error
        elif db_key == 'ink_price_per_kg': fallback_val = FALLBACK_INK_PRICE
        elif db_key == 'varnish_price_per_kg': fallback_val = FALLBACK_VARNISH_PRICE
        elif db_key == 'machine_labor_price_per_hour': fallback_val = FALLBACK_LABOR_PRICE
        elif db_key == 'tool_price_semirotary': fallback_val = FALLBACK_TOOL_SEMI_PRICE
        elif db_key == 'tool_price_rotary': fallback_val = FALLBACK_TOOL_ROT_PRICE
        elif db_key == 'plate_price_per_color': fallback_val = FALLBACK_PLATE_PRICE
        elif db_key == 'machine_speed_default': fallback_val = FALLBACK_MACHINE_SPEED
        else: fallback_val = min_val if min_val is not None else 0.0

        # Try loading from DB settings (st.session_state.settings should exist by now)
        settings_dict = st.session_state.get('settings', {})
        loaded_value = settings_dict.get(db_key, fallback_val)
        st.session_state[state_key] = loaded_value
        logger.debug(f"Initialized state '{state_key}' from settings/fallback: {loaded_value} (DB key: {db_key})")

    # Get current value from state (should be initialized by now)
    # Ensure state value is not None before passing to widget, use fallback if it is
    current_val_from_state = st.session_state.get(state_key)
    if current_val_from_state is None:
        logger.warning(f"State key '{state_key}' was None, using min_val or 0.0 for widget.")
        current_val_from_state = min_val if min_val is not None else 0.0
        st.session_state[state_key] = current_val_from_state # Update state as well

    # Create the number input widget, passing the disabled argument
    input_val = st.sidebar.number_input(
        label,
        min_value=min_val,
        max_value=max_val,
        value=current_val_from_state, # Use the validated state value
        step=step,
        format=format_str,
        key=widget_key, # Use the unique widget key
        help=help_text,
        disabled=disabled  # Pass disabled argument to widget
    )

    # Check if the user changed the value in the widget (only if not disabled)
    needs_update = False
    if not disabled: # Only check for updates if the widget is enabled
        if isinstance(input_val, float) and isinstance(current_val_from_state, (float, int)):
            if not math.isclose(input_val, float(current_val_from_state), rel_tol=1e-7, abs_tol=1e-9):
                needs_update = True
        elif isinstance(input_val, int) and isinstance(current_val_from_state, (float, int)):
             # Handle potential type mismatch (e.g., state is float 30.0, input becomes int 30)
             if not math.isclose(float(input_val), float(current_val_from_state), rel_tol=1e-7, abs_tol=1e-9):
                 needs_update = True
        elif input_val != current_val_from_state: # Fallback for other types or initial load mismatch
            needs_update = True

    if needs_update:
        logger.info(f"Detected change in '{label}' (Widget Key: {widget_key}): {current_val_from_state} -> {input_val}")
        st.session_state[state_key] = input_val # Update session state first
        logger.debug(f"Updating state '{state_key}' to: {input_val}")

        # Attempt to update the database
        if update_setting_in_db(db_key, input_val):
            logger.info(f"Successfully updated DB for key '{db_key}' with value {input_val}")
            # Caches are cleared within update_setting_in_db now
        else:
            st.sidebar.error(f"Failed to update {label} in DB!", icon="💾")
            logger.error(f"Failed DB update for key '{db_key}' with value {input_val}")
            # Consider reverting state if DB update fails critically
            # st.session_state[state_key] = current_val_from_state

    # Return the current value (from state, which reflects the latest user input or DB load)
    return st.session_state[state_key]


# --- Database Initialization on Startup ---
# Use flags to ensure DB init and data loading run only once per session start
if 'db_initialized_flag' not in st.session_state:
    st.session_state.db_initialized_flag = False
    st.session_state.db_init_success = False # Track success status

if not st.session_state.db_initialized_flag:
    logger.info("Running database initialization check...")
    with st.spinner("Connecting to database and verifying schema..."):
        db_init_success = init_db() # Attempt initialization
    st.session_state.db_init_success = db_init_success
    st.session_state.db_initialized_flag = True # Mark init as attempted
    if db_init_success:
        logger.info("Database initialization reported successful.")
    else:
        logger.error("Database initialization reported failed.")
        st.error("CRITICAL: Database initialization failed. Check logs. Using fallback values.", icon="🚨")

# --- Session State Initialization for Data ---
if 'data_loaded_flag' not in st.session_state:
    st.session_state.data_loaded_flag = False

# Load data only if DB init was successful and data hasn't been loaded yet
if st.session_state.db_init_success and not st.session_state.data_loaded_flag:
    logger.info("DB init successful, proceeding to load data into session state...")
    conn = get_db_connection() # Get the connection for cache keys
    if conn:
        # Load settings and materials from DB (using cache)
        st.session_state.settings = load_settings_from_db(conn)
        st.session_state.materials_prices = load_materials_from_db(conn)

        # Handle potential loading failures
        if st.session_state.settings is None:
            st.session_state.settings = {} # Ensure settings is a dict even if load fails
            st.error("Failed to load settings from database! Using fallback values.", icon="⚙️")
            logger.error("load_settings_from_db returned None.")
        else:
            logger.info(f"Successfully loaded/cached {len(st.session_state.settings)} settings.")

        if st.session_state.materials_prices is None:
            st.session_state.materials_prices = {} # Ensure materials is a dict even if load fails
            st.error("Failed to load materials from database! Using fallback values.", icon="📄")
            logger.error("load_materials_from_db returned None.")
             # Provide minimal fallback if materials are crucial and load failed
            if not st.session_state.materials_prices:
                 st.session_state.materials_prices = {"Paper (chrome)": 39.95}
                 logger.warning("Materials list was empty after failed load, added fallback material.")
        else:
             logger.info(f"Successfully loaded/cached {len(st.session_state.materials_prices)} materials.")

        # Initialize individual state keys needed elsewhere, using loaded settings/fallbacks
        # Note: synced_number_input will primarily rely on these being present later
        st.session_state.ink_price_per_kg = st.session_state.settings.get('ink_price_per_kg', FALLBACK_INK_PRICE)
        st.session_state.varnish_price_per_kg = st.session_state.settings.get('varnish_price_per_kg', FALLBACK_VARNISH_PRICE)
        st.session_state.machine_labor_price_per_hour = st.session_state.settings.get('machine_labor_price_per_hour', FALLBACK_LABOR_PRICE)
        st.session_state.tool_price_semirotary = st.session_state.settings.get('tool_price_semirotary', FALLBACK_TOOL_SEMI_PRICE)
        st.session_state.tool_price_rotary = st.session_state.settings.get('tool_price_rotary', FALLBACK_TOOL_ROT_PRICE)
        st.session_state.plate_price_per_color = st.session_state.settings.get('plate_price_per_color', FALLBACK_PLATE_PRICE)
        st.session_state.machine_speed_default = int(st.session_state.settings.get('machine_speed_default', FALLBACK_MACHINE_SPEED))
        st.session_state.single_calc_profit_coefficient = st.session_state.settings.get('single_calc_profit_coefficient', FALLBACK_SINGLE_PROFIT)
        # Initialize profit coefficients for offer quantities
        for qty in QUANTITIES_FOR_OFFER:
            key = f"profit_coeff_{qty}"
            fallback = FALLBACK_PROFITS.get(qty, 0.20) # Use specific fallback for quantity
            st.session_state[key] = st.session_state.settings.get(key, fallback)
            # logger.debug(f"Initialized state '{key}' from settings/fallback: {st.session_state[key]}")

        # Initialize other non-setting state variables if they don't exist
        if 'client_name_input' not in st.session_state: st.session_state.client_name_input = ""
        if 'product_name_input' not in st.session_state: st.session_state.product_name_input = ""
        if 'template_width_input' not in st.session_state: st.session_state.template_width_input = 76.0
        if 'template_height_input' not in st.session_state: st.session_state.template_height_input = 76.0
        if 'quantity_input' not in st.session_state: st.session_state.quantity_input = 10000
        if 'is_blank_check' not in st.session_state: st.session_state.is_blank_check = False
        if 'num_colors_select' not in st.session_state: st.session_state.num_colors_select = 1
        if 'is_uv_varnish_check' not in st.session_state: st.session_state.is_uv_varnish_check = False
        if 'machine_speed_slider' not in st.session_state: st.session_state.machine_speed_slider = st.session_state.machine_speed_default
        if 'tool_type_radio' not in st.session_state: st.session_state.tool_type_radio = "None"
        if 'existing_tool_info' not in st.session_state: st.session_state.existing_tool_info = ""
        if 'material_select' not in st.session_state: st.session_state.material_select = None # Will be set based on list later
        if 'offer_results_list' not in st.session_state: st.session_state.offer_results_list = []
        if 'offer_pdf_buffer' not in st.session_state: st.session_state.offer_pdf_buffer = None
        if 'show_history_check_state' not in st.session_state: st.session_state.show_history_check_state = False
        if 'search_client_input' not in st.session_state: st.session_state.search_client_input = ""
        if 'search_product_input' not in st.session_state: st.session_state.search_product_input = ""
        if 'search_results_df' not in st.session_state: st.session_state.search_results_df = pd.DataFrame()
        if 'load_calc_select' not in st.session_state: st.session_state.load_calc_select = None


        st.session_state.data_loaded_flag = True # Mark data as loaded
        logger.info("Session state initialized with data.")

    else:
         logger.error("Failed to get DB connection for data loading.")
         st.error("Failed to establish database connection for loading data.", icon="🔌")
         # Set flags to prevent continuous attempts if connection fails critically
         st.session_state.db_init_success = False
         st.session_state.data_loaded_flag = True # Mark as 'done' trying to load

# Handle the case where DB init failed initially - ensure minimal fallbacks exist
elif not st.session_state.db_init_success and st.session_state.db_initialized_flag:
     logger.warning("DB initialization previously failed. Skipping data loading from DB, ensuring fallbacks.")
     # Ensure essential state variables exist with fallbacks if DB init failed
     if 'settings' not in st.session_state: st.session_state.settings = {}
     if 'materials_prices' not in st.session_state: st.session_state.materials_prices = {"Paper (chrome)": 39.95}
     # Initialize other state variables using FALLBACKs
     st.session_state.ink_price_per_kg = FALLBACK_INK_PRICE
     st.session_state.varnish_price_per_kg = FALLBACK_VARNISH_PRICE
     st.session_state.machine_labor_price_per_hour = FALLBACK_LABOR_PRICE
     st.session_state.tool_price_semirotary = FALLBACK_TOOL_SEMI_PRICE
     st.session_state.tool_price_rotary = FALLBACK_TOOL_ROT_PRICE
     st.session_state.plate_price_per_color = FALLBACK_PLATE_PRICE
     st.session_state.machine_speed_default = FALLBACK_MACHINE_SPEED
     st.session_state.single_calc_profit_coefficient = FALLBACK_SINGLE_PROFIT
     for qty in QUANTITIES_FOR_OFFER:
         key = f"profit_coeff_{qty}"
         st.session_state[key] = FALLBACK_PROFITS.get(qty, 0.20)
     # Initialize UI state variables as well
     if 'client_name_input' not in st.session_state: st.session_state.client_name_input = ""
     # ... (add all other UI state variables from the success block above) ...
     if 'existing_tool_info' not in st.session_state: st.session_state.existing_tool_info = ""
     if 'offer_results_list' not in st.session_state: st.session_state.offer_results_list = []
     if 'offer_pdf_buffer' not in st.session_state: st.session_state.offer_pdf_buffer = None
     if 'show_history_check_state' not in st.session_state: st.session_state.show_history_check_state = False
     if 'search_results_df' not in st.session_state: st.session_state.search_results_df = pd.DataFrame()
     # ...etc


# --- Streamlit Application UI ---
st.title("📊 Label Printing Cost Calculator & Offer Generator")
st.markdown("Enter parameters in the sidebar. Calculation is shown below. Generate a multi-quantity offer or search history at the bottom.")

# Stop execution if DB initialization fundamentally failed
if not st.session_state.get('db_init_success', False):
    # Error message already shown during init attempt
    st.stop() # Halt execution

# Check if essential data dictionaries are loaded (even if fallbacks were used)
if 'settings' not in st.session_state or 'materials_prices' not in st.session_state:
     st.error("Essential settings or materials data could not be initialized. Application might not function correctly.", icon="⚠️")
     # Decide if you want to st.stop() here depending on criticality
     # st.stop()

# --- Sidebar ---
st.sidebar.header("Input Parameters")
# Use session state for text inputs to preserve values during load/rerun
st.session_state.client_name_input = st.sidebar.text_input("Client Name:", value=st.session_state.get('client_name_input', ''), key="client_name_widget")
st.session_state.product_name_input = st.sidebar.text_input("Product/Label Name:", value=st.session_state.get('product_name_input', ''), key="product_name_widget")
st.sidebar.markdown("---")
# Use session state for number inputs
st.session_state.template_width_input = st.sidebar.number_input("Template Width (W, mm):", min_value=0.1, value=st.session_state.get('template_width_input', 76.0), step=0.1, format="%.3f", key="template_width_widget")
st.session_state.template_height_input = st.sidebar.number_input("Template Height (H, mm):", min_value=0.1, value=st.session_state.get('template_height_input', 76.0), step=0.1, format="%.3f", key="template_height_widget")
st.session_state.quantity_input = st.sidebar.number_input("Desired Quantity (single calc):", min_value=1, value=st.session_state.get('quantity_input', 10000), step=1000, format="%d", key="quantity_widget")
st.sidebar.markdown("---")

st.sidebar.subheader("Ink, Varnish, and Plate Settings")
# Use session state for checkboxes
st.session_state.is_blank_check = st.sidebar.checkbox("Blank Template (no ink)", value=st.session_state.get('is_blank_check', False), key="is_blank_widget")
st.session_state.num_colors_select = st.sidebar.number_input("Number of Colors:", min_value=1, max_value=8, value=st.session_state.get('num_colors_select', 1), step=1, format="%d", disabled=st.session_state.is_blank_check, key="num_colors_widget")
st.session_state.is_uv_varnish_check = st.sidebar.checkbox("UV Varnish", value=st.session_state.get('is_uv_varnish_check', False), help=f"Assumes {GRAMS_VARNISH_PER_M2}g/m² consumption", key="is_uv_varnish_widget")

# Use synced_number_input for settings managed in the DB
ink_price_kg_input = synced_number_input("Ink Price (RSD/kg):", 'ink_price_per_kg', 'ink_price_per_kg', step=10.0, format_str="%.2f", disabled=st.session_state.is_blank_check)
varnish_price_kg_input = synced_number_input("UV Varnish Price (RSD/kg):", 'varnish_price_per_kg', 'varnish_price_per_kg', step=10.0, format_str="%.2f", disabled=not st.session_state.is_uv_varnish_check)
plate_price_input = synced_number_input("Plate Price per Color (RSD):", 'plate_price_per_color', 'plate_price_per_color', step=50.0, format_str="%.2f", disabled=st.session_state.is_blank_check)
st.sidebar.markdown("---")

st.sidebar.subheader("Machine")
# Use session state for slider
machine_speed_default_val = st.session_state.get('machine_speed_default', FALLBACK_MACHINE_SPEED)
st.session_state.machine_speed_slider = st.sidebar.slider("Average Machine Speed (m/min):", MACHINE_SPEED_MIN, MACHINE_SPEED_MAX, st.session_state.get('machine_speed_slider', machine_speed_default_val), 5, key="machine_speed_widget")
labor_price_h_input = synced_number_input("Machine Labor Price (RSD/h):", 'machine_labor_price_per_hour', 'machine_labor_price_per_hour', step=50.0, format_str="%.2f")
st.sidebar.markdown("---")

st.sidebar.subheader("Cutting Tool")
tool_type_options_keys = ["None", "Semirotary", "Rotary"]
# Use session state for radio button
tool_key_from_state = st.session_state.get('tool_type_radio', "None")
tool_index = tool_type_options_keys.index(tool_key_from_state) if tool_key_from_state in tool_type_options_keys else 0
st.session_state.tool_type_radio = st.sidebar.radio(
    "Select tool type:",
    options=tool_type_options_keys,
    index=tool_index,
    key="tool_type_widget"
    )
selected_tool_key = st.session_state.tool_type_radio # Get value from state after widget render

# Input for existing tool ID, only shown if "None" is selected
# Use session state to preserve the input value across reruns
if selected_tool_key == "None":
    st.session_state.existing_tool_info = st.sidebar.text_input(
        "Existing tool ID/Name (if any):",
        value=st.session_state.get('existing_tool_info', ""), # Get from state or default empty
        help="Enter identifier if using an existing tool (no new tool cost).",
        key="existing_tool_widget" # Unique key for this widget
        )
existing_tool_info_input = st.session_state.get('existing_tool_info', "") # Use the value from state

# Use synced_number_input for tool prices (always show, but cost applied based on selection)
tool_price_semi_input = synced_number_input("Semirotary Tool Price (RSD):", 'tool_price_semirotary', 'tool_price_semirotary', step=100.0, format_str="%.0f")
tool_price_rot_input = synced_number_input("Rotary Tool Price (RSD):", 'tool_price_rotary', 'tool_price_rotary', step=100.0, format_str="%.0f")
st.sidebar.markdown("---")


st.sidebar.subheader("Material")
# Ensure materials_prices exists and is a dictionary
materials_dict = st.session_state.get('materials_prices', {})
if not isinstance(materials_dict, dict):
    logger.warning(f"materials_prices in session state is not a dict: {materials_dict}. Resetting.")
    materials_dict = {"Paper (chrome)": 39.95} # Fallback
    st.session_state.materials_prices = materials_dict

material_list = list(materials_dict.keys())
selected_material = None # Initialize
price_per_m2_input = None # Initialize

if not material_list:
    st.sidebar.error("No materials found!", icon="⚠️")
    st.sidebar.warning("Add a material below to proceed.")
    # selected_material remains None
    price_per_m2_input = 0.0 # Default price input if no material can be selected
else:
    # Try to keep the previously selected material if it still exists
    previous_selection = st.session_state.get("material_select", None)
    default_index = 0
    if previous_selection and previous_selection in material_list:
        try:
            default_index = material_list.index(previous_selection)
        except ValueError:
             logger.warning(f"Previously selected material '{previous_selection}' no longer in list.")
             default_index = 0 # Fallback to first item
    elif material_list:
        default_index = 0 # Default to first item if no previous or invalid previous
    else:
         default_index = -1 # Should not happen if material_list is not empty

    if default_index != -1:
        # Use session state for selectbox
        st.session_state.material_select = st.sidebar.selectbox(
            "Select material type:",
            options=material_list,
            index=default_index,
            key="material_select_widget" # Use key to preserve selection
        )
        selected_material = st.session_state.material_select # Get current selection

        # Get current price from state/dict for the selected material
        current_material_price_state = materials_dict.get(selected_material, 0.0)

        material_price_label_formatted = f"Price for '{selected_material}' (RSD/m²):"
        # Use a unique key for the price input widget based on material name
        price_input_key = f"material_price_input_{selected_material.replace(' ', '_').replace('(', '').replace(')', '').lower()}"

        # Use session state for material price input
        # Initialize if not present for this specific key
        if price_input_key not in st.session_state:
             st.session_state[price_input_key] = current_material_price_state

        st.session_state[price_input_key] = st.sidebar.number_input(
            material_price_label_formatted,
            min_value=0.0,
            value=st.session_state[price_input_key], # Use state value as default
            step=0.1,
            format="%.2f",
            key=f"{price_input_key}_widget", # Unique key for widget itself
            disabled=(selected_material is None)
        )
        price_per_m2_input = st.session_state[price_input_key] # Get current value from state


        # Check if the price was changed by the user for the *selected* material
        if selected_material and not math.isclose(price_per_m2_input, current_material_price_state, rel_tol=1e-7, abs_tol=1e-9):
            logger.info(f"Detected change in price for '{selected_material}': {current_material_price_state} -> {price_per_m2_input}")
            # Update the dictionary in session state FIRST
            st.session_state.materials_prices[selected_material] = price_per_m2_input
            # Attempt to update the database
            if update_material_price_in_db(selected_material, price_per_m2_input):
                logger.info(f"Successfully updated DB price for '{selected_material}'. Rerunning.")
                st.rerun() # Rerun to reflect the change immediately
            else:
                st.sidebar.error(f"Failed to update price for {selected_material} in DB.", icon="💾")
                # Revert state change if DB fails
                st.session_state.materials_prices[selected_material] = current_material_price_state
                st.session_state[price_input_key] = current_material_price_state
                st.rerun() # Rerun to show reverted value


    else:
        # Case where material list became empty unexpectedly
        st.sidebar.error("Material selection error.", icon="⁉️")
        selected_material = None
        price_per_m2_input = 0.0

# Expander for adding new material
with st.sidebar.expander("➕ Add New Material"):
    new_material_name = st.text_input("New Material Name", key="new_mat_name_widget").strip()
    new_material_price = st.number_input("New Material Price (RSD/m²)", min_value=0.01, step=0.1, format="%.2f", key="new_mat_price_widget")
    if st.button("Add Material", key="add_mat_button"):
        if new_material_name and new_material_price > 0:
            # Check if material already exists (case-insensitive check recommended)
            existing_names = [name.lower() for name in st.session_state.get('materials_prices', {}).keys()]
            if new_material_name.lower() in existing_names:
                st.sidebar.warning(f"Material '{new_material_name}' already exists (case-insensitive).", icon="❗")
            else:
                logger.info(f"Attempting to add new material: '{new_material_name}' with price {new_material_price}")
                success, _ = execute_with_retry(
                    "INSERT INTO materials (name, price_per_m2) VALUES (?, ?)",
                    (new_material_name, new_material_price)
                )
                if success:
                    st.sidebar.success(f"Material '{new_material_name}' added!", icon="✅")
                    logger.info(f"Successfully added material '{new_material_name}'. Reloading and rerunning.")
                    # Update state directly and clear cache
                    conn = get_db_connection()
                    if conn:
                        load_materials_from_db.clear() # Clear cache
                        st.session_state.materials_prices = load_materials_from_db(conn) # Reload
                    st.rerun() # Rerun to update the selectbox
                else:
                    st.sidebar.error(f"Failed to add material '{new_material_name}'. Check logs.", icon="❌")
                    logger.error(f"Failed DB insert for new material '{new_material_name}'.")
        else:
            st.sidebar.warning("Please enter both a unique name and a price > 0.", icon="❗")
st.sidebar.markdown("---")


st.sidebar.subheader("Profit Coefficient (Single Calc)")
st.sidebar.caption("Used only for the calculation shown at the top.")
# Use synced_number_input for single profit coefficient
single_calc_profit_input = synced_number_input(
    label="Profit Coeff:", state_key='single_calc_profit_coefficient', db_key='single_calc_profit_coefficient',
    min_val=0.00, step=0.01, format_str="%.3f", help_text="Profit margin as a decimal (e.g., 0.20 for 20%) applied to material cost.", is_profit_coeff=True
)
st.sidebar.markdown("---")


st.sidebar.subheader("Profit Coefficients (Offer)")
st.sidebar.caption("Used only when generating the multi-quantity offer.")
profit_coeffs_inputs = {}
# Use synced_number_input for each offer quantity profit coefficient
for qty in sorted(QUANTITIES_FOR_OFFER): # Sort quantities for display order
    state_key = f'profit_coeff_{qty}'
    db_key = f'profit_coeff_{qty}'
    profit_coeffs_inputs[qty] = synced_number_input(
        label=f"Coeff. for {qty:,}:", state_key=state_key, db_key=db_key,
        min_val=0.00, step=0.01, format_str="%.3f", help_text=f"Profit coefficient for quantity {qty:,}", is_profit_coeff=True
    )


# --- Main Calculation & Display Area ---
st.header("📊 Calculation Results (Single Quantity)")

# --- Retrieve values from state for calculation ---
# These values should now reflect user input or loaded data
client_name = st.session_state.get('client_name_input', '')
product_name = st.session_state.get('product_name_input', '')
template_width_W_input = st.session_state.get('template_width_input', 0.0)
template_height_H_input = st.session_state.get('template_height_input', 0.0)
quantity_input = st.session_state.get('quantity_input', 0)
is_blank = st.session_state.get('is_blank_check', False)
num_colors_input = st.session_state.get('num_colors_select', 1)
is_uv_varnish_input = st.session_state.get('is_uv_varnish_check', False)
machine_speed_m_min = st.session_state.get('machine_speed_slider', FALLBACK_MACHINE_SPEED)
selected_tool_key = st.session_state.get('tool_type_radio', "None")
existing_tool_info_input = st.session_state.get('existing_tool_info', "") if selected_tool_key == "None" else ""
selected_material = st.session_state.get('material_select', None)
# price_per_m2_input is derived within the material selection logic

# --- Input Validation ---
inputs_valid = True
error_messages = []

if not (template_width_W_input and template_width_W_input > 0):
    inputs_valid = False; error_messages.append("Template Width (W) must be > 0.")
if not (template_height_H_input and template_height_H_input > 0):
    inputs_valid = False; error_messages.append("Template Height (H) must be > 0.")
if not (quantity_input and quantity_input > 0):
    inputs_valid = False; error_messages.append("Desired Quantity must be > 0.")
if not is_blank and not (num_colors_input and num_colors_input >= 1):
    # Allow 0 colors if blank is checked later? Let's be strict for now.
    inputs_valid = False; error_messages.append("Number of Colors must be >= 1 if not blank.")
if not selected_material:
    inputs_valid = False; error_messages.append("Material must be selected.")
# Price_per_m2_input could be 0, which is valid, check None explicitly if derived logic could fail
if price_per_m2_input is None and selected_material: # Check if price failed to load for selected material
    inputs_valid = False; error_messages.append(f"Could not determine price for material '{selected_material}'.")
elif price_per_m2_input is None and not selected_material:
     pass # Handled by "Material must be selected"
elif price_per_m2_input < 0:
     inputs_valid = False; error_messages.append("Material price cannot be negative.")

if not (machine_speed_m_min and machine_speed_m_min >= MACHINE_SPEED_MIN):
    inputs_valid = False; error_messages.append(f"Machine Speed must be >= {MACHINE_SPEED_MIN}.")
# Check required settings from session state (should be loaded by synced_number_input)
required_state_keys = [
    'ink_price_per_kg', 'varnish_price_per_kg', 'plate_price_per_color',
    'machine_labor_price_per_hour', 'tool_price_semirotary', 'tool_price_rotary',
    'single_calc_profit_coefficient'
]
for key in required_state_keys:
    if key not in st.session_state or st.session_state[key] is None:
         inputs_valid = False; error_messages.append(f"Setting '{key}' is missing or invalid (check DB/fallbacks).")
    elif isinstance(st.session_state[key], (int, float)) and st.session_state[key] < 0:
         inputs_valid = False; error_messages.append(f"Setting '{key}' cannot be negative.")


# --- Run Calculation if Inputs are Valid ---
pdf_buffer = None
calculation_data_for_db = {} # Data dict for saving and PDF generation
single_calc_result = {} # Result dict from the calculation function
best_circumference_solution = None
number_across_width_y = 0
tool_info_string_display = "" # For display/saving
current_calc_params = {} # To store params used for the current single calc


if inputs_valid:
    logger.info("Inputs deemed valid, proceeding with single calculation.")
    # 1. Find Cylinder Specs
    best_circumference_solution, all_circumference_solutions, circumference_message = find_cylinder_specifications(template_width_W_input)

    if best_circumference_solution:
        logger.debug(f"Best cylinder solution found: {best_circumference_solution}")
        # 2. Calculate Number Across Width
        number_across_width_y = calculate_number_across_width(template_height_H_input, WORKING_WIDTH, WIDTH_GAP)
        logger.debug(f"Calculated number across width (y): {number_across_width_y}")

        # 3. Prepare Parameters for Calculation Function
        # Get the single profit coefficient from state
        single_profit_coeff = st.session_state.get('single_calc_profit_coefficient')
        if single_profit_coeff is None: # Should not happen if validation passed, but check again
             logger.error("Single profit coefficient is None unexpectedly!")
             st.error("Critical error: Profit coefficient missing.", icon="🔥")
             single_profit_coeff = 0.0 # Use safe default

        # Determine number of colors for calculation (0 if blank)
        valid_num_colors_for_calc = 0 if is_blank else num_colors_input

        current_calc_params = {
            "quantity": quantity_input,
            "template_width_W": template_width_W_input,
            "template_height_H": template_height_H_input,
            "best_circumference_solution": best_circumference_solution,
            "number_across_width_y": number_across_width_y,
            "is_blank": is_blank,
            "num_colors": valid_num_colors_for_calc,
            "is_uv_varnish": is_uv_varnish_input,
            "price_per_m2": price_per_m2_input, # Price for the selected material
            "machine_speed_m_min": machine_speed_m_min,
            "selected_tool_key": selected_tool_key, # Key like "None", "Semirotary", "Rotary"
            "existing_tool_info": existing_tool_info_input, # Text field value if key is "None"
            "profit_coefficient": single_profit_coeff,
            # Get other prices from session state (validated earlier)
            "ink_price_kg": st.session_state.ink_price_per_kg,
            "varnish_price_kg": st.session_state.varnish_price_per_kg,
            "plate_price_color": st.session_state.plate_price_per_color,
            "labor_price_hour": st.session_state.machine_labor_price_per_hour,
            "tool_price_semi": st.session_state.tool_price_semirotary,
            "tool_price_rot": st.session_state.tool_price_rotary
        }

        # 4. Run the Calculation
        single_calc_result = run_single_calculation(**current_calc_params)

        # 5. Process Results
        if 'error' not in single_calc_result:
            logger.info("Single calculation successful.")
            # Prepare data for display, DB saving, and PDF generation
            tool_info_string_display = single_calc_result.get('tool_info_string_final', 'N/A') # Get final string from results

            calculation_data_for_db = {
                # Input parameters snapshot
                "client_name": client_name,
                "product_name": product_name,
                "template_width_W_input": template_width_W_input,
                "template_height_H_input": template_height_H_input,
                "quantity_input": quantity_input,
                "is_blank": is_blank,
                "valid_num_colors_for_calc": valid_num_colors_for_calc,
                "is_uv_varnish_input": is_uv_varnish_input,
                "selected_material": selected_material,
                "price_per_m2": price_per_m2_input, # Include material price used
                "tool_info_string_final": tool_info_string_display, # Save the descriptive string (used by save_calc_to_db)
                "machine_speed_m_min": machine_speed_m_min,
                # Calculation details & results
                "best_circumference_solution": best_circumference_solution,
                # Note: gap_G_circumference_mm is now added inside run_single_calculation results
                "number_circumference_x": best_circumference_solution.get('templates_N_circumference', 0),
                "number_across_width_y": number_across_width_y,
                 # Add all results from single_calc_result dictionary
                **single_calc_result
            }

            # --- Display Results ---
            st.subheader(f"Details for Qty: {quantity_input:,} pcs (Profit Coeff: {single_profit_coeff:.3f})")

            # Expander for detailed breakdown
            with st.expander("Calculation Details (Config, Consumption, Time)"):
                # Parameter Summary String (condensed)
                params_dims = f"W:{template_width_W_input:.2f}×H:{template_height_H_input:.2f}mm"
                params_qty = f"Qty:{quantity_input:,}"
                params_colors = 'Blank' if is_blank else f"{valid_num_colors_for_calc}C"
                params_varnish = '+V' if is_uv_varnish_input else ''
                params_mat = f"Mat:'{selected_material}'({price_per_m2_input:.2f}RSD/m²)"
                params_tool = f"Tool:'{tool_info_string_display}'" # Use final string
                params_speed = f"Speed:{machine_speed_m_min:.0f}m/min"
                params_profit = f"Prof.Coef:{single_profit_coeff:.3f}"
                st.caption(f"{params_dims} | {params_qty} | {params_colors}{params_varnish} | {params_mat} | {params_tool} | {params_speed} | {params_profit}")
                st.markdown("---")

                # 1. Cylinder & Template Config
                st.subheader("1. Cylinder & Template")
                col1_cfg, col2_cfg = st.columns(2)
                with col1_cfg:
                    st.metric("Teeth (Z)", f"{best_circumference_solution.get('number_of_teeth_Z', 'N/A')}")
                    st.metric("Circumference", f"{best_circumference_solution.get('circumference_mm', 0.0):.3f} mm")
                    st.metric("Circumference Gap (G)", f"{single_calc_result.get('gap_G_circumference_mm', 'N/A'):.3f} mm") # From calculation results
                with col2_cfg:
                    st.metric("Templates Circumference (x)", f"{best_circumference_solution.get('templates_N_circumference', 'N/A')}")
                    st.metric("Templates Width (y)", f"{number_across_width_y}")
                    st.metric("Format (y × x)", f"{number_across_width_y} × {best_circumference_solution.get('templates_N_circumference', 'N/A')}")

                # 2. Material Width
                st.subheader("2. Material Width")
                if number_across_width_y > 0:
                    mat_col1a, mat_col2a = st.columns([2,1])
                    # Construct help string for width calculation
                    gap_count = max(0, number_across_width_y - 1)
                    help_width_a = f"({number_across_width_y}×{template_height_H_input:.2f}) + ({gap_count}×{WIDTH_GAP}) + {WIDTH_WASTE}"
                    req_w = single_calc_result.get('required_material_width_mm', 0)
                    with mat_col1a:
                        st.metric("Required Width", f"{req_w:.2f} mm", help=help_width_a)
                    with mat_col2a:
                        if not single_calc_result.get('material_width_exceeded'):
                            st.success(f"✅ OK (≤ {MAX_MATERIAL_WIDTH} mm)")
                        else:
                            st.error(f"⚠️ EXCEEDED! (> {MAX_MATERIAL_WIDTH} mm)")
                else:
                    st.warning("Number across width (y) is 0, width not applicable.")

                # 3/4. Material Consumption
                st.subheader("3/4. Material Consumption (Production + Waste)")
                tot_len = single_calc_result.get('total_final_length_m', 0)
                tot_area = single_calc_result.get('total_final_area_m2', 0)
                prod_len = single_calc_result.get('total_production_length_m', 0)
                waste_len = single_calc_result.get('waste_length_m', 0)
                tot_col1a, tot_col2a = st.columns(2)
                with tot_col1a:
                    st.metric("TOTAL Length", f"{tot_len:,.2f} m", help=f"Production: {prod_len:,.1f}m + Waste: {waste_len:,.1f}m")
                with tot_col2a:
                    st.metric("TOTAL Area", f"{tot_area:,.2f} m²")

                # 5. Estimated Production Time
                st.subheader("5. Estimated Production Time")
                time_col1a, time_col2a, time_col3a, time_col4a = st.columns(4)
                t_setup = single_calc_result.get('setup_time_min', 0)
                t_prod = single_calc_result.get('production_time_min', 0)
                t_clean = single_calc_result.get('cleanup_time_min', 0)
                t_total = single_calc_result.get('total_time_min', 0)
                with time_col1a: st.metric("Setup", format_time(t_setup))
                with time_col2a: st.metric("Production", format_time(t_prod))
                with time_col3a: st.metric("Cleanup", format_time(t_clean))
                with time_col4a: st.metric("TOTAL", format_time(t_total))

            # --- Costs & Final Price Display (Outside Expander) ---
            st.markdown("---")
            st.subheader(f"💰 Costs & Final Price for Qty: {quantity_input:,}")

            cost_cols = st.columns(6)
            cost_cols[0].metric("Ink", f"{single_calc_result.get('ink_cost_rsd', 0):,.2f} RSD")
            cost_cols[1].metric("Varnish", f"{single_calc_result.get('varnish_cost_rsd', 0):,.2f} RSD")
            cost_cols[2].metric("Plates", f"{single_calc_result.get('plate_cost_rsd', 0):,.2f} RSD")
            cost_cols[3].metric("Material", f"{single_calc_result.get('material_cost_rsd', 0):,.2f} RSD")
            cost_cols[4].metric("Tool", f"{single_calc_result.get('tool_cost_rsd', 0):,.2f} RSD", help=tool_info_string_display)
            cost_cols[5].metric("Labor", f"{single_calc_result.get('labor_cost_rsd', 0):,.2f} RSD")

            st.markdown("---") # Separator before totals

            price_cols = st.columns(3)
            total_prod_cost = single_calc_result.get('total_production_cost_rsd', 0)
            profit_value = single_calc_result.get('profit_rsd', 0)
            profit_coeff_used = single_calc_result.get('profit_coefficient_used')
            profit_delta_str = f"{profit_coeff_used*100:.1f}%" if profit_coeff_used is not None else None

            price_cols[0].metric("Total Prod. Cost", f"{total_prod_cost:,.2f} RSD")
            price_cols[1].metric("Profit", f"{profit_value:,.2f} RSD", delta=profit_delta_str, help="Based on material cost")
            price_cols[2].metric("TOTAL SELLING PRICE", f"{single_calc_result.get('total_selling_price_rsd', 0):,.2f} RSD")

            st.metric("Selling Price / Piece", f"{single_calc_result.get('selling_price_per_piece_rsd', 0):.4f} RSD")

            # --- Generate Calculation PDF ---
            with st.spinner("Generating calculation PDF..."):
                pdf_buffer = create_pdf(calculation_data_for_db)
            if pdf_buffer is None:
                 st.warning("Could not generate calculation PDF.", icon="📄")


        else:
            # Handle calculation error from run_single_calculation
            error_msg = single_calc_result.get('error', 'Unknown calculation error')
            st.error(f"❌ Calculation failed for Qty {quantity_input:,}: {error_msg}")
            logger.error(f"run_single_calculation failed: {error_msg}")

    else:
        # Handle failure to find cylinder specs
        error_msg = circumference_message or "Circumference calculation failed (no valid solutions found)."
        st.error(f"❌ Cannot proceed with calculation: {error_msg}")
        logger.error(f"find_cylinder_specifications failed: {error_msg}")

else:
    # Inputs are not valid, display collected error messages
    st.warning("Please correct the input errors in the sidebar:", icon="⚠️")
    for msg in error_messages:
        st.warning(f"- {msg}")


# --- Offer Generation Section ---
st.markdown("---")
st.header("📋 Offer Generation (Multiple Quantities)")

# Enable preview button only if the single calculation was successful
offer_button_disabled = not (inputs_valid and best_circumference_solution and current_calc_params and 'error' not in single_calc_result)

if st.button("🔄 Preview/Update Offer Prices", key="preview_offer_button", disabled=offer_button_disabled, help="Calculates prices for standard quantities using current parameters and offer profit coefficients."):
    if not offer_button_disabled:
        temp_offer_results_preview = []
        total_quantities = len(QUANTITIES_FOR_OFFER)
        progress_text = f"Calculating offer prices... {{i}}/{total_quantities}" # Use f-string later
        progress_bar_preview = st.progress(0, text=progress_text.format(i=0))

        with st.spinner("Calculating offer prices..."):
            # Use the parameters from the successful single calculation as a base
            base_params_for_offer = current_calc_params.copy()
            calculation_succeeded_for_all = True

            for i, qty in enumerate(sorted(QUANTITIES_FOR_OFFER)): # Calculate in sorted order
                offer_params = base_params_for_offer.copy()
                offer_params["quantity"] = qty # Update quantity

                # Get the specific profit coefficient for this quantity from state
                coeff_key = f'profit_coeff_{qty}'
                specific_profit_coeff = st.session_state.get(coeff_key)
                if specific_profit_coeff is None:
                     logger.error(f"Profit coefficient for qty {qty} (key {coeff_key}) not found in session state!")
                     st.warning(f"Profit coefficient for Qty {qty:,} not found, using fallback.", icon="⚠️")
                     specific_profit_coeff = FALLBACK_PROFITS.get(qty, 0.20) # Fallback

                offer_params["profit_coefficient"] = specific_profit_coeff
                logger.debug(f"Running offer calc for Qty: {qty:,} with Profit Coeff: {specific_profit_coeff:.3f}")

                # Run calculation for this quantity
                result = run_single_calculation(**offer_params)

                # Update progress bar
                progress_bar_preview.progress((i + 1) / total_quantities, text=progress_text.format(i=i+1))


                if 'error' not in result:
                    temp_offer_results_preview.append({
                        "Količina (kom)": qty,
                        "Cena/kom (RSD)": result.get('selling_price_per_piece_rsd', 0),
                        "Ukupno (RSD)": result.get('total_selling_price_rsd', 0),
                        "Profit Coeff Used": specific_profit_coeff # Store for reference
                    })
                else:
                    st.warning(f"Calculation failed for Offer Qty {qty:,}: {result['error']}", icon="❌")
                    logger.warning(f"Offer calculation failed for Qty {qty:,}: {result['error']}")
                    calculation_succeeded_for_all = False # Mark that at least one failed

            progress_bar_preview.empty() # Remove progress bar

            if temp_offer_results_preview:
                st.session_state.offer_results_list = temp_offer_results_preview
                st.session_state.offer_pdf_buffer = None # Clear old offer PDF buffer
                logger.info(f"Offer price preview generated with {len(temp_offer_results_preview)} results.")
            else:
                st.warning("Offer price preview could not be generated (all calculations failed?).", icon="📉")
                st.session_state.offer_results_list = [] # Clear any previous results

            if not calculation_succeeded_for_all:
                 st.warning("Some quantities in the offer failed to calculate. Results shown are partial.", icon="⚠️")


    else:
        st.warning("Cannot generate offer preview. Ensure a successful single calculation is performed first with valid inputs.", icon="🚫")


# --- Display Offer Preview Table ---
if st.session_state.get('offer_results_list'):
    st.subheader("Offer Summary Preview")
    st.write(f"**Client:** {client_name if client_name else '(Not specified)'}")
    st.write(f"**Product:** {product_name if product_name else '(Not specified)'}")

    st.write("**Specifications:**")
    valid_num_colors_for_calc = 0 if is_blank else num_colors_input # Recalculate for display
    num_colors_display_offer = 'Blank' if is_blank else valid_num_colors_for_calc
    spec_data_offer = {
        "Dimenzija (mm)": f"{template_width_W_input:.2f} x {template_height_H_input:.2f}",
        "Materijal": selected_material,
        "Broj boja": num_colors_display_offer,
        "UV Lak": "Da" if is_uv_varnish_input else "Ne",
        "Alat": tool_info_string_display # Use the string derived from calculation
    }
    spec_df_offer = pd.DataFrame(spec_data_offer.items(), columns=['Stavka', 'Vrednost'])
    st.dataframe(spec_df_offer, hide_index=True, use_container_width=True)
    st.write("**Prices per Quantity (Preview):**")

    offer_df_display = pd.DataFrame(st.session_state.offer_results_list)
    offer_df_display['Količina (kom)'] = offer_df_display['Količina (kom)'].map('{:,}'.format)
    offer_df_display['Cena/kom (RSD)'] = offer_df_display['Cena/kom (RSD)'].map('{:.4f}'.format)
    offer_df_display['Ukupno (RSD)'] = offer_df_display['Ukupno (RSD)'].map('{:,.2f}'.format)
    offer_df_display['Profit Coeff Used'] = offer_df_display['Profit Coeff Used'].map('{:.3f}'.format)

    st.dataframe(offer_df_display, hide_index=True, use_container_width=True)
else:
    st.info("Click 'Preview/Update Offer Prices' above to calculate and display the offer table based on current inputs and offer profit coefficients.", icon="👆")


# --- Action Buttons ---
st.markdown("---")
action_cols = st.columns(3)

# 1. Download Calculation PDF Button
with action_cols[0]:
    pdf_calc_download_disabled = pdf_buffer is None
    if not pdf_calc_download_disabled:
        safe_product_name = "".join(c if c.isalnum() else "_" for c in product_name) if product_name else "product"
        safe_client_name = "".join(c if c.isalnum() else "_" for c in client_name) if client_name else "client"
        pdf_calc_filename = f"Calc_{safe_product_name}_{safe_client_name}_{quantity_input}pcs_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button(
            label="📄 Download Calc PDF", data=pdf_buffer, file_name=pdf_calc_filename, mime="application/pdf",
            key="pdf_calc_download", use_container_width=True, disabled=pdf_calc_download_disabled,
            help="Downloads a detailed PDF report for the single quantity calculation shown above."
        )
    else:
        st.button("📄 Download Calc PDF", disabled=True, use_container_width=True, help="Requires a successful single calculation first.")

# 2. Save Calculation to DB Button
with action_cols[1]:
    # Enable saving only if calculation was successful and data is available
    save_disabled = not calculation_data_for_db or 'error' in single_calc_result
    if st.button("💾 Save Calc to DB", disabled=save_disabled, key="save_calc_button", use_container_width=True, help="Saves the results of the single quantity calculation to the history database."):
        if calculation_data_for_db:
            with st.spinner("Saving calculation..."):
                if save_calculation_to_db(calculation_data_for_db):
                    st.success("Calculation saved to DB!", icon="💾")
                    # Optionally clear history cache if implemented
        else:
             st.warning("No valid calculation data available to save.", icon="🚫")


# 3. Generate Final Offer PDF Button & Download
with action_cols[2]:
    generate_offer_pdf_disabled = not st.session_state.get('offer_results_list')
    if st.button("📝 Generate Offer PDF", key="generate_final_offer_button", disabled=generate_offer_pdf_disabled, use_container_width=True, help="Generates the final Offer PDF based on the previewed prices."):
        if st.session_state.offer_results_list and current_calc_params: # Ensure base params exist too
            with st.spinner("Generating final offer PDF..."):
                # Prepare data for the offer PDF function
                valid_num_colors_for_calc_offer = 0 if is_blank else num_colors_input
                num_colors_display_pdf = 'Blank' if is_blank else valid_num_colors_for_calc_offer
                spec_data_pdf = {
                    "Dimenzija (mm)": f"{template_width_W_input:.2f} x {template_height_H_input:.2f}",
                    "Materijal": selected_material,
                    "Broj boja": num_colors_display_pdf,
                    "UV Lak": "Da" if is_uv_varnish_input else "Ne",
                    "Alat": tool_info_string_display # Use the final string
                }
                offer_pdf_data = {
                    "client_name": client_name,
                    "product_name": product_name,
                    "specifications": spec_data_pdf,
                    "offer_results": st.session_state.offer_results_list # Use the previewed data
                }
                pdf_gen_buffer = create_offer_pdf(offer_pdf_data)

                if pdf_gen_buffer:
                    st.session_state.offer_pdf_buffer = pdf_gen_buffer # Store in state for download
                    st.success("Final Offer PDF generated!", icon="✅")
                    st.rerun() # Rerun to make download button available immediately
                else:
                    st.error("Failed to generate final offer PDF.", icon="❌")
                    st.session_state.offer_pdf_buffer = None # Ensure buffer is None on failure
        else:
            st.warning("No offer results available to generate PDF. Please Preview/Update first.", icon="📋")

    # Offer PDF Download Button (conditionally displayed)
    offer_pdf_download_disabled = st.session_state.get('offer_pdf_buffer') is None
    if not offer_pdf_download_disabled:
        safe_product_name_offer = "".join(c if c.isalnum() else "_" for c in product_name) if product_name else "product"
        safe_client_name_offer = "".join(c if c.isalnum() else "_" for c in client_name) if client_name else "client"
        offer_pdf_filename = f"Offer_{safe_product_name_offer}_{safe_client_name_offer}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button(
            label="⬇️ Download Offer PDF", data=st.session_state.offer_pdf_buffer, file_name=offer_pdf_filename,
            mime="application/pdf", key="pdf_offer_download_final", use_container_width=True,
            help="Downloads the generated multi-quantity Offer PDF."
        )
    else:
        st.button("⬇️ Download Offer PDF", disabled=True, use_container_width=True, help="Generate the Offer PDF first using the button above.")


# --- Search and Load History Section ---
st.markdown("---")
with st.expander("🔍 Search and Load Calculation History"):
    search_col1, search_col2 = st.columns(2)
    with search_col1:
        # Use session state for search inputs
        st.session_state.search_client_input = st.text_input("Search by Client Name:", value=st.session_state.get('search_client_input', ''), key="search_client_widget")
    with search_col2:
        st.session_state.search_product_input = st.text_input("Search by Product Name:", value=st.session_state.get('search_product_input', ''), key="search_product_widget")

    if st.button("Search History", key="search_history_button"):
        # Retrieve search terms from state
        client_query = st.session_state.search_client_input
        product_query = st.session_state.search_product_input
        # Run search and save results to state
        with st.spinner("Searching history..."):
             results_df = search_calculations(client_query=client_query, product_query=product_query)
        if results_df is not None:
            st.session_state.search_results_df = results_df
            st.session_state.load_calc_select = None # Reset load selection on new search
        else:
            # Error message already shown by search_calculations
            st.session_state.search_results_df = pd.DataFrame() # Empty DF on error

    # Display search results if they exist in state
    search_results_df = st.session_state.get('search_results_df')
    if search_results_df is not None and not search_results_df.empty:
        st.write("Search Results:")
        # Format TotalPrice for display
        search_results_df_display = search_results_df.copy()
        if 'TotalPrice' in search_results_df_display.columns:
             search_results_df_display['TotalPrice'] = search_results_df_display['TotalPrice'].map('{:,.2f}'.format)
        st.dataframe(search_results_df_display, use_container_width=True, hide_index=True)

        # --- Load Functionality ---
        st.markdown("---")
        st.write("Load Calculation Parameters:")

        # Create list of options for selectbox from search results
        load_options_list = [
            f"ID: {row['id']} - {row['Client']} - {row['Product']} (Qty: {row['Qty']:,})"
            for index, row in search_results_df.iterrows()
        ]

        if load_options_list:
            # Use session state for selectbox selection
            selected_option = st.selectbox(
                "Select calculation to load parameters from:",
                options=load_options_list,
                index=None, # Start with no selection
                placeholder="Choose a calculation...",
                key="load_calc_select" # State key for the selectbox itself
            )
            st.session_state.load_calc_select_value = selected_option # Store the chosen value

            if selected_option:
                # Extract ID from selected option
                try:
                    selected_id_str = selected_option.split(" - ")[0].split(": ")[1]
                    selected_id = int(selected_id_str)
                except (IndexError, ValueError) as e:
                    st.error("Could not extract ID from selected option.", icon="❌")
                    logger.error(f"Error parsing selected_id from '{selected_option}': {e}")
                    selected_id = None

                if selected_id is not None:
                     load_button_key = f"load_button_{selected_id}"
                     if st.button(f"Load Parameters from ID: {selected_id}", key=load_button_key):
                        logger.info(f"Attempting to load parameters for calculation ID: {selected_id}")
                        with st.spinner(f"Loading data for ID {selected_id}..."):
                            loaded_data = load_calculation_details(selected_id)

                        if loaded_data:
                            try:
                                # --- UPDATE SESSION STATE ---
                                logger.debug(f"Applying loaded data: {loaded_data}")

                                # Simple Inputs (update state keys tied to widgets)
                                st.session_state.client_name_input = loaded_data.get('client_name', '')
                                st.session_state.product_name_input = loaded_data.get('product_name', '')
                                st.session_state.template_width_input = loaded_data.get('template_width', 0.0)
                                st.session_state.template_height_input = loaded_data.get('template_height', 0.0)
                                st.session_state.quantity_input = loaded_data.get('quantity', 1000)

                                # Checkboxes
                                st.session_state.is_blank_check = loaded_data.get('is_blank', False)
                                st.session_state.is_uv_varnish_check = loaded_data.get('is_uv_varnish', False)

                                # Number inputs (non-synced)
                                loaded_colors = loaded_data.get('num_colors', 1)
                                st.session_state.num_colors_select = loaded_colors if not st.session_state.is_blank_check else 1

                                # Slider
                                st.session_state.machine_speed_slider = int(loaded_data.get('machine_speed', FALLBACK_MACHINE_SPEED))

                                # Synced inputs: ONLY load the single profit coeff used for that specific calculation.
                                # Other synced inputs (prices) should reflect CURRENT settings, not old ones.
                                if 'profit_coefficient' in loaded_data and loaded_data['profit_coefficient'] is not None:
                                     st.session_state.single_calc_profit_coefficient = loaded_data['profit_coefficient']
                                     logger.info(f"Loaded single profit coefficient from history: {loaded_data['profit_coefficient']:.3f}")
                                else:
                                     # If not saved, maybe revert to current default? Or keep as is.
                                     st.session_state.single_calc_profit_coefficient = st.session_state.settings.get('single_calc_profit_coefficient', FALLBACK_SINGLE_PROFIT)
                                     logger.info("Profit coefficient not found in loaded data, using current default.")


                                # Material
                                loaded_material_name = loaded_data.get('material_name')
                                if loaded_material_name:
                                    current_materials_dict = st.session_state.get('materials_prices', {})
                                    if loaded_material_name in current_materials_dict:
                                        st.session_state.material_select = loaded_material_name
                                        # Update price input widget state to current price
                                        current_price = current_materials_dict[loaded_material_name]
                                        price_input_key_loaded = f"material_price_input_{loaded_material_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}"
                                        st.session_state[price_input_key_loaded] = current_price
                                        logger.info(f"Loaded material '{loaded_material_name}' selection.")
                                    else:
                                        st.warning(f"Material '{loaded_material_name}' from saved calc no longer exists. Select manually.", icon="⚠️")
                                        logger.warning(f"Loaded material '{loaded_material_name}' not found in current materials: {list(current_materials_dict.keys())}")
                                        st.session_state.material_select = None # Reset selection
                                else:
                                     st.warning("Saved calculation did not contain a material name.", icon="⚠️")
                                     st.session_state.material_select = None # Reset selection

                                # Tool
                                loaded_tool_info = loaded_data.get('tool_type', "None") # Get saved string
                                logger.debug(f"Loading tool info string: '{loaded_tool_info}'")
                                tool_key_to_set = "None"
                                existing_tool_id_to_set = ""

                                if loaded_tool_info:
                                    if loaded_tool_info.startswith("Semirotary"): tool_key_to_set = "Semirotary"
                                    elif loaded_tool_info.startswith("Rotary"): tool_key_to_set = "Rotary"
                                    elif loaded_tool_info.startswith("Existing: "):
                                        tool_key_to_set = "None"
                                        try: existing_tool_id_to_set = loaded_tool_info.split("Existing: ", 1)[1].strip()
                                        except IndexError: existing_tool_id_to_set = ""
                                    elif loaded_tool_info == "None (No Tool Cost)" or loaded_tool_info == "None":
                                         tool_key_to_set = "None"
                                    else: # Unknown string, default to None
                                         logger.warning(f"Unknown tool type string loaded: '{loaded_tool_info}', defaulting to None.")
                                         tool_key_to_set = "None"

                                st.session_state.tool_type_radio = tool_key_to_set # Update radio button state
                                st.session_state.existing_tool_info = existing_tool_id_to_set # Update text input state

                                logger.info(f"Loaded tool selection: Key='{tool_key_to_set}', ExistingID='{existing_tool_id_to_set}'")

                                st.success(f"Successfully loaded parameters from calculation ID: {selected_id}", icon="✅")
                                logger.info(f"Parameter loading complete for ID: {selected_id}. Rerunning.")

                                # Clear search results and selection to prevent accidental re-load
                                st.session_state.search_results_df = pd.DataFrame()
                                st.session_state.load_calc_select = None
                                st.rerun() # Rerun script to reflect loaded values in UI

                            except KeyError as ke:
                                 st.error(f"Error applying loaded parameters: Missing expected key '{ke}' in loaded data.", icon="❌")
                                 logger.error(f"KeyError applying loaded parameters for ID {selected_id}: {ke}", exc_info=True)
                            except Exception as e:
                                st.error(f"Error applying loaded parameters: {e}", icon="❌")
                                logger.error(f"Error applying loaded parameters for ID {selected_id}: {e}", exc_info=True)
                        else:
                             # Error message already shown by load_calculation_details
                             pass
        else:
            # This condition met if search results exist but load_options_list is somehow empty
            st.info("No results available in the format needed for loading.")


    elif 'search_results_df' in st.session_state and search_results_df is not None and search_results_df.empty:
        # Only show this if a search was performed and returned empty
        st.info("No calculations found matching your search criteria.")
    # else: # Initial state before any search
    #     st.info("Enter search criteria and click 'Search History'.")


# --- History Display (Original Checkbox) ---
st.markdown("---") # Separator before the simple history display
st.subheader("📜 Calculation History (Last 10)")
show_history = st.checkbox("Show History", value=st.session_state.get('show_history_check_state', False), key="show_history_widget")
st.session_state.show_history_check_state = show_history # Update state

if show_history:
    # @st.cache_data # Consider caching history view if DB is slow
    def load_history_display():
        logger.debug("Loading last 10 calculations for display.")
        query = """
            SELECT
                strftime('%Y-%m-%d %H:%M', timestamp) as Timestamp,
                client_name as Client,
                product_name as Product,
                quantity as Qty,
                template_width as W,
                template_height as H,
                num_colors as Colors,
                is_blank as Blank,
                material_name as Material,
                tool_type as Tool,
                profit_coefficient as ProfitCoeff,
                calculated_price_per_piece as PricePerPc,
                calculated_total_price as TotalPrice
            FROM calculations
            ORDER BY timestamp DESC
            LIMIT 10
        """
        success, result = execute_with_retry(query)
        if success:
            if result:
                history_data = [dict(row) for row in result]
                history_df = pd.DataFrame(history_data)
                # Format for display
                history_df['Blank'] = history_df['Blank'].apply(lambda x: 'Yes' if x else 'No')
                history_df['PricePerPc'] = history_df['PricePerPc'].map('{:.4f}'.format)
                history_df['TotalPrice'] = history_df['TotalPrice'].map('{:,.2f}'.format)
                history_df['ProfitCoeff'] = history_df['ProfitCoeff'].map('{:.3f}'.format)
                history_df['W'] = history_df['W'].map('{:.2f}'.format)
                history_df['H'] = history_df['H'].map('{:.2f}'.format)
                return history_df
            else:
                logger.info("Calculation history (last 10) is empty.")
                return pd.DataFrame() # Return empty DataFrame
        else:
            logger.error("Failed to load calculation history (last 10) from DB.")
            st.error("Error loading history data.", icon="💾")
            return None # Indicate error

    history_df_display = load_history_display()

    if history_df_display is not None:
        if not history_df_display.empty:
            st.dataframe(history_df_display, use_container_width=True, hide_index=True)
        else:
            st.info("No calculations saved in the history yet.")


# --- Footer ---
st.markdown("---")
try:
    # Retrieve current values from state for footer display
    footer_labor = st.session_state.get('machine_labor_price_per_hour', 'N/A')
    footer_ink = st.session_state.get('ink_price_per_kg', 'N/A')
    footer_plate = st.session_state.get('plate_price_per_color', 'N/A')
    # Safely format numbers
    footer_labor_str = f"{footer_labor:,.0f}" if isinstance(footer_labor, (int, float)) else str(footer_labor)
    footer_ink_str = f"{footer_ink:,.0f}" if isinstance(footer_ink, (int, float)) else str(footer_ink)
    footer_plate_str = f"{footer_plate:,.0f}" if isinstance(footer_plate, (int, float)) else str(footer_plate)

    settings_footer_str = (f"Current Base Prices (RSD): Labor/h: {footer_labor_str} | Ink/kg: {footer_ink_str} | Plate/color: {footer_plate_str}")
    st.caption(settings_footer_str)
except Exception as e:
    logger.warning(f"Could not format footer string: {e}")
    st.caption("Could not display current settings.")

logger.info("--- Streamlit script execution finished ---")
