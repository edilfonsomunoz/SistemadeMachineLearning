# Guía de Despliegue en Render.com

Esta aplicación está lista para desplegarse en Render.com de manera simple y gratuita.

## 📋 Requisitos Previos

1. Cuenta en [Render.com](https://render.com) (gratis)
2. Código subido a GitHub/GitLab
3. Archivo `requirements.txt` incluido (ya creado)

## 🚀 Pasos para Desplegar

### 1. Preparar el Repositorio

Asegúrate de que tu proyecto esté en GitHub. Si usas Replit:
- Ve a la pestaña "Version control" 
- Conecta con GitHub
- Sube (push) tus cambios

### 2. Crear Web Service en Render

1. Inicia sesión en [Render.com](https://dashboard.render.com)
2. Haz clic en **"New +"** → **"Web Service"**
3. Conecta tu repositorio de GitHub/GitLab
4. Selecciona el repositorio de este proyecto

### 3. Configurar el Servicio

En la página de configuración, usa estos valores:

**Información Básica:**
- **Name**: `ml-system` (o el nombre que prefieras)
- **Region**: Elige la más cercana a ti
- **Branch**: `main` (o tu rama principal)
- **Runtime**: `Python 3`

**Build & Deploy:**
- **Build Command**: 
  ```bash
  pip install -r requirements.txt
  ```

- **Start Command**: 
  ```bash
  gunicorn --bind 0.0.0.0:$PORT app:app
  ```

**Plan:**
- Selecciona **"Free"** (gratis, suficiente para empezar)

### 4. Variables de Entorno (Opcional)

Solo si necesitas activar el modo debug (NO RECOMENDADO en producción):

- **Key**: `DEBUG`
- **Value**: `false` (o déjalo vacío para producción)

### 5. Desplegar

1. Haz clic en **"Create Web Service"**
2. Render automáticamente:
   - Descargará tu código
   - Instalará las dependencias
   - Creará la base de datos SQLite
   - Iniciará la aplicación

### 6. Acceder a tu Aplicación

Después de unos minutos, verás:
- ✅ Estado: "Live"
- 🌐 URL pública: `https://tu-app.onrender.com`

¡Listo! Tu aplicación de Machine Learning está en línea.

## 📊 Base de Datos

La aplicación usa **SQLite** (archivo local), por lo que:

✅ **Ventajas:**
- No requiere configuración de base de datos externa
- Completamente gratis
- Datos persisten entre despliegues

⚠️ **Limitación:**
- En el plan gratuito, Render puede borrar datos después de inactividad
- Para datos permanentes, considera usar el plan de pago

## 🔄 Actualizaciones Automáticas

Render redespliegará automáticamente cuando:
- Hagas `git push` a tu rama principal
- O actives un redespliegue manual desde el dashboard

## 🛠️ Solución de Problemas

**Error al instalar dependencias:**
- Verifica que `requirements.txt` esté en la raíz del proyecto

**La app no inicia:**
- Revisa los logs en el dashboard de Render
- Verifica que el comando de inicio sea correcto

**Base de datos vacía después de redespliegue:**
- Es normal en el plan gratuito después de inactividad
- Los usuarios deberán registrarse nuevamente

## 📚 Recursos Adicionales

- [Documentación de Render](https://render.com/docs)
- [Render Free Tier](https://render.com/docs/free)
- [Troubleshooting Guide](https://render.com/docs/troubleshooting)

---

## 🎓 Notas para Estudiantes

Esta es una aplicación educativa. Para uso en producción real:
1. Considera usar una base de datos PostgreSQL externa
2. Implementa límites de tasa (rate limiting)
3. Configura CORS con orígenes específicos
4. Añade monitoreo y logging profesional
