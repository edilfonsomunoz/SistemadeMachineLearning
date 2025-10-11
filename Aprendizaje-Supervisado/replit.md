# Overview

This is a Machine Learning analysis system that combines a Flask backend with a Vite-powered frontend. The application allows users to register, upload datasets, run various ML models (regression, logistic regression, and decision trees), and generate PDF reports of their analyses. The system is designed for educational purposes, targeting students who need to perform ML experiments and document their results.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture

**Technology Stack**: The frontend uses Vite as the build tool with vanilla HTML, CSS, and JavaScript (no framework). This provides fast development with hot module reloading (HMR) enabled by default.

**Design Pattern**: Single Page Application (SPA) with client-side routing handled through DOM manipulation and visibility toggling. Authentication state is managed through localStorage with JWT tokens.

**Key Components**:
- Authentication UI (login/register forms)
- Dashboard for dataset management
- Model execution interface with dynamic form generation
- Results visualization displaying charts and metrics

**Rationale**: Vanilla JavaScript was chosen for simplicity and educational clarity, avoiding framework complexity. Vite provides modern development experience with fast refresh while maintaining simplicity.

## Backend Architecture

**Framework**: Flask with Flask-CORS for API endpoints. The backend follows a simple REST API pattern without complex routing frameworks.

**Authentication**: Session-based authentication using bcrypt for password hashing and custom token generation with secrets module. Tokens are stored client-side and validated server-side.

**ML Pipeline**: Modular architecture with separate model classes:
- `RegressionModels`: Linear, polynomial, RBF, Ridge, and Lasso regression
- `LogisticModels`: Binary and multi-class logistic regression
- `DecisionTreeModels`: ID3, C4.5, and CART implementations
- `DataHandler`: Centralized data preparation and validation
- `ReportGenerator`: PDF generation using ReportLab

**Rationale**: The modular ML architecture allows easy addition of new models. Each model class is self-contained with its own training, prediction, and visualization logic. Flask was chosen for its simplicity and Python's strong ML ecosystem compatibility.

## Data Flow

1. User uploads CSV/Excel files through the frontend
2. Backend stores file metadata and links to user session
3. User selects model type and parameters
4. Backend processes data through scikit-learn pipelines
5. Results (metrics, visualizations) are returned as JSON
6. PDF reports are generated on-demand with embedded charts

**Data Handling**: All numerical features are automatically selected and missing values are filled with column means. The system auto-detects classification vs regression tasks based on target variable characteristics.

## Report Generation

Uses ReportLab to generate professional PDF reports containing:
- User session information
- Model configurations and parameters
- Performance metrics and evaluation results
- Base64-encoded matplotlib visualizations
- Formatted tables for coefficients and feature importance

**Rationale**: PDF generation on the server side ensures consistent formatting and allows inclusion of complex visualizations without client-side rendering issues.

# External Dependencies

## Backend Dependencies

**Core Framework**:
- `Flask`: Web framework for API endpoints
- `Flask-CORS`: Cross-origin resource sharing for frontend-backend communication

**Database**:
- `psycopg2`: PostgreSQL database adapter
- PostgreSQL connection via `DATABASE_URL` environment variable
- Uses `RealDictCursor` for JSON-friendly query results

**Machine Learning**:
- `scikit-learn`: Core ML algorithms (LinearRegression, LogisticRegression, DecisionTree, etc.)
- `pandas`: Data manipulation and CSV/Excel file handling
- `numpy`: Numerical computations

**Security**:
- `bcrypt`: Password hashing for secure authentication
- `python-dotenv`: Environment variable management

**Visualization & Reporting**:
- `matplotlib`: Chart generation (configured with 'Agg' backend for server-side rendering)
- `ReportLab`: PDF report generation with custom styling

## Frontend Dependencies

**Build Tools**:
- `Vite ^5.4.8`: Development server and build tool with HMR support

**Runtime**: Pure vanilla JavaScript with no frontend frameworks or libraries. All functionality is implemented using native Web APIs.

## Database Schema

**Users Table**:
- `id`: Serial primary key
- `email`: Unique user identifier
- `name`: User's full name
- `student_code`: 6-digit student identifier
- `password_hash`: Bcrypt-hashed password
- `created_at`: Registration timestamp

**Sessions Table**: (Implied from code) Stores dataset metadata and model execution history linked to user accounts

## Environment Configuration

Required environment variables:
- `DATABASE_URL`: PostgreSQL connection string for database access
- `DEBUG`: Optional, set to "true" to enable Flask debug mode (default: False)

The application expects a PostgreSQL database to be available and will initialize required tables on first run.

# Security Hardening (October 2025)

The application has been hardened against common web vulnerabilities:

## Security Measures Implemented

1. **Debug Mode Protection**
   - Debug mode disabled by default (`debug=False`)
   - Only activatable via DEBUG environment variable
   - Prevents exposure of sensitive stack traces in production

2. **Secure Error Handling**
   - All API endpoints use generic error messages for client responses
   - Detailed error information logged server-side with `logging.error(..., exc_info=True)`
   - No stack traces or internal exception details exposed to clients
   - Prevents information disclosure attacks

3. **Static File Isolation**
   - Frontend files served from dedicated `static/` directory
   - API routes segregated under `/api/*` prefix
   - Backend source code (app.py, ml_models/) not accessible via HTTP
   - Prevents unauthorized access to application source code

4. **Authentication Security**
   - Password hashing with bcrypt (industry-standard)
   - Session tokens generated with `secrets.token_urlsafe(32)`
   - Token-based authentication with 7-day expiration
   - Database-backed session management

## Recommended Future Enhancements

1. **CORS Configuration**: Configure CORS with explicit origin whitelist before production deployment
2. **Session Cleanup**: Implement automated cleanup of expired sessions
3. **Rate Limiting**: Add rate limiting to prevent brute force attacks
4. **Input Validation**: Enhanced input sanitization for file uploads