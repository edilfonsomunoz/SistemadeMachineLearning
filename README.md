# ML Dashboard - Streamlit Application

## Overview

This is an interactive web application built with Streamlit that provides a comprehensive Machine Learning dashboard. The application allows users to upload datasets, preprocess data, train various ML models (regression, classification, decision trees), compare model performance, tune hyperparameters, and export results. It includes user authentication with PostgreSQL for session persistence and analysis history tracking.

## Recent Changes

### October 12, 2025
- **UI Enhancement**: Centered login and registration forms in styled boxes for better visual presentation
- **Visualization Enhancement**: Decision tree visualization now always displays regardless of tree depth, with dynamic sizing based on actual depth:
  - Shallow trees (depth ≤3): Clear visualization with larger fonts
  - Medium trees (depth 4-5): Standard visualization
  - Deep trees (depth 6-8): Larger canvas with informational message
  - Very deep trees (depth >8): Extra large canvas with warning message
- **Security Enhancement**: Migrated password hashing from SHA-256 to bcrypt with automatic salt generation for improved security against rainbow table attacks
- **Bug Fix**: Fixed comparison dashboard crash when only classification models exist (no regression models). The dashboard now correctly identifies the best classification model as the overall best model when no regression models are present
- **Robustness**: Added null-safety checks to prevent crashes when displaying best model metrics

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework**: Streamlit  
- **Rationale**: Chosen for rapid development of data science applications with minimal frontend code. Streamlit provides built-in components for file uploads, data visualization, and interactive widgets that align perfectly with ML workflow requirements.
- **Page Structure**: Multi-page application with sidebar navigation and tab-based interfaces within each module
- **State Management**: Uses Streamlit's session_state to persist user authentication status, datasets, and model results across page interactions

### Backend Architecture

**Modular Design**:
- **Separation of Concerns**: Application is split into distinct modules (auth, preprocessing, models, reports, etc.) where each handles a specific domain
- **Session Management**: User sessions and analysis history are persisted to enable workflow continuity
- **Data Processing Pipeline**: Sequential flow from data upload → preprocessing → model training → comparison → export

**Key Components**:

1. **Authentication System** (`auth.py`)
   - Password hashing with bcrypt for security
   - Email validation using regex patterns
   - Student code validation (6-digit requirement)
   - PostgreSQL-based user storage

2. **Data Processing** (`preprocesamiento.py`)
   - File upload support for CSV and Excel formats
   - Null value handling with mean imputation
   - Categorical encoding using LabelEncoder
   - Data quality checks and validation

3. **ML Model Implementation** (`modelos.py`, `hyperparameter_tuning.py`)
   - Regression models: Linear, Ridge, Lasso, Polynomial, Kernel Ridge (RBF)
   - Classification models: Logistic Regression, Decision Trees
   - Automated train-test splitting
   - Grid Search with cross-validation for hyperparameter optimization
   - Comprehensive metrics calculation

4. **Model Comparison** (`comparacion_modelos.py`)
   - Cross-model performance analysis
   - Visualization of metrics across different model types
   - Best model identification and ranking

5. **Reporting & Export** (`reportes.py`, `exportar_datos.py`)
   - PDF report generation using FPDF
   - CSV/Excel export for processed datasets and predictions
   - Metrics export functionality

6. **Session History** (`session_history.py`)
   - Analysis session persistence
   - Historical session retrieval
   - JSON serialization for complex model results

### Data Storage Solutions

**Primary Database**: PostgreSQL
- **Rationale**: Chosen for robust relational data storage with ACID compliance. Required for multi-user authentication and session management.
- **Schema Design**:
  - `users` table: Stores user credentials (email, name, student_code, hashed_password, timestamps)
  - `analysis_sessions` table: Stores analysis sessions with JSON fields for dataset_info and model_results
- **Connection Pattern**: Environment variable-based connection string (`DATABASE_URL`) for deployment flexibility

**File-Based Storage**: 
- Legacy JSON file (`usuarios.json`) present but appears to be replaced by PostgreSQL implementation
- Temporary file storage for uploaded datasets during session

### Authentication & Authorization

**Authentication Flow**:
1. User registration with validation (email format, 6-digit student code, password requirements)
2. Password hashing with bcrypt using salt for security
3. Session-based authentication stored in Streamlit session_state
4. Database verification on login with password hash comparison

**Security Measures**:
- Bcrypt password hashing with salt (not plain text storage)
- SQL parameterized queries to prevent injection attacks
- Email validation using regex patterns
- Session state isolation per user

### Key Design Patterns

1. **Session State Pattern**: Centralized state management using Streamlit's session_state for authentication status, loaded datasets, and model results

2. **Module Separation**: Each ML workflow step (preprocessing, modeling, comparison, export) is isolated in separate modules for maintainability

3. **Error Handling**: Database connection errors are caught and user-friendly messages displayed via Streamlit's UI components

4. **Visualization Pipeline**: Matplotlib/Seaborn for chart generation → Streamlit for rendering → FPDF for report export

## External Dependencies

### Core Framework
- **Streamlit**: Web application framework for the entire UI and interaction layer

### Machine Learning & Data Science
- **scikit-learn**: All ML algorithms (LinearRegression, Ridge, Lasso, LogisticRegression, DecisionTree, RandomForest, GridSearchCV)
- **pandas**: DataFrame operations and data manipulation
- **numpy**: Numerical computations and array operations

### Visualization
- **matplotlib**: Primary plotting library for charts and graphs
- **seaborn**: Statistical visualizations and enhanced plotting aesthetics

### Database & Storage
- **psycopg2**: PostgreSQL database adapter for Python
- **Environment Variable**: `DATABASE_URL` - PostgreSQL connection string

### Security
- **bcrypt**: Password hashing and verification

### Reporting
- **FPDF**: PDF generation for analysis reports

### Data Processing
- **sklearn.preprocessing**: LabelEncoder for categorical encoding, StandardScaler for normalization
- **File Format Support**: CSV and Excel (xlsx, xls) via pandas

### Utilities
- **re** (regex): Email validation patterns
- **json**: Serialization for complex data structures in database
- **io**: In-memory file operations for exports
- **datetime**: Timestamp generation for sessions
- **warnings**: Suppression of sklearn deprecation warnings
