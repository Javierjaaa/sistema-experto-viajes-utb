# Sistema Experto · Destinos Turísticos de Bolivia

Proyecto académico de **Inteligencia Artificial**: un sistema experto basado en
reglas de producción (IF-THEN) que recomienda los **dos** destinos turísticos de
Bolivia más afines a las preferencias del usuario.

## Componentes

| Componente | Implementación |
| --- | --- |
| Hechos | 6 variables del formulario (motivación, actividad, social, alojamiento, duración, clima) |
| Base de conocimiento | 37 destinos de Bolivia modelados como reglas `IF ... AND ... THEN ...` en `BASE_CONOCIMIENTO` |
| Motor de inferencia | Red bayesiana ingenua: nodo oculto `Destino` (37 estados) y 6 nodos observados; inferencia exacta en log-espacio, Top 2 |
| Backend | Flask (`app.py`) |
| Frontend | `templates/index.html` con Tailwind CSS desde CDN |

## Requisitos

- Python 3.10 o superior
- pip

## Ejecución en Windows (ya configurado)

Haz **doble clic en `iniciar.bat`**. El script activa el entorno virtual
(incluido en la carpeta `venv/`), abre el navegador y levanta el servidor en
**http://127.0.0.1:5000**.

## Instalación y ejecución manual

```bash
# 1. (Opcional) crear entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 2. Instalar dependencias
pip install flask

# 3. Ejecutar la aplicación
python app.py
```

Abre el navegador en **http://127.0.0.1:5000**

> También puedes instalar desde el archivo incluido: `pip install -r requirements.txt`

## Estructura del proyecto

```
sistema-experto-viajes/
├── app.py               # Backend, hechos, base de conocimiento y motor de inferencia
├── templates/
│   └── index.html       # Formulario y resultados (Tailwind CSS)
├── iniciar.bat          # Arranque con un doble clic (Windows)
├── venv/                # Entorno virtual con Flask instalado
├── requirements.txt
└── README.md
```

## Interfaz

- Barra de **Progreso del formulario** que se completa al elegir las 6 variables.
- Columnas **Preferencias Personales** y **Preferencias de Viaje**.
- Botón **Obtener Recomendación** habilitado solo al 100 % del formulario.
- Tarjeta **Recomendación Lista** con el destino principal y la alternativa,
  su puntaje de coincidencias, las reglas que se activaron, la duración ideal,
  la altitud/clima, las actividades y un enlace externo de información.
- Botón **Nueva Consulta** para reiniciar el ciclo.

## ¿Cómo funciona el motor de inferencia?

El motor es una **red bayesiana ingenua** (`RedBayesiana` en `app.py`):

1. El formulario envía los **hechos** (una respuesta por cada variable).
2. `leer_hechos()` valida que cada valor pertenezca al universo de discurso.
3. El nodo oculto `Destino` tiene 37 estados (uno por regla) con **prior
   uniforme** (`1/37`). De él dependen los 6 nodos observados (motivación,
   actividad, social, alojamiento, duración y clima).
4. Cada tabla condicional se deriva de la regla:
   `P(valor canónico | destino) = 0,75` y el 0,25 restante se reparte por igual
   entre las otras opciones de esa variable (`PROB_COINCIDENCIA`).
5. `inferir()` aplica el teorema de Bayes en **log-espacio** para evitar
   underflow: `P(destino | hechos) ∝ P(destino) · Π P(valorᵢ | destino)`,
   y normaliza para que las 37 probabilidades sumen 1.
6. `evaluar_regla()` cuenta las coincidencias solo para mostrar las reglas
   activadas; `motor_inferencia()` ordena por **probabilidad bayesiana** y
   devuelve las **Top 2** reglas.
7. La confianza se calcula sobre esa probabilidad: **Alta** ≥ 35 %,
   **Media** ≥ 15 %, **Baja** < 15 %.
8. Si ninguna regla coincide (puntaje 0), el sistema informa que no hay
   recomendaciones válidas para esa combinación.

### Catálogo de destinos (37 reglas)

Cada destino incluye: nombre, departamento, descripción, duración ideal,
altitud/clima, actividades, enlace a BoliviaTurismo y sus 6 condiciones.
Las reglas R21–R37 cubren todas las páginas de "Destinos y ciudades" del sitio
que aún no estaban representadas.

| ID | Destino | Departamento | Duración ideal | Condiciones (6/6) |
| --- | --- | --- | --- | --- |
| R01 | Salar de Uyuni | Potosí | 3 a 5 días | Aventura · Alta · Media · Rústico · Mediano · Frío |
| R02 | Lago Titicaca e Isla del Sol | La Paz | 2 a 3 días | Cultura · Moderada · Media · Tradicional · Corto · Frío |
| R03 | Parque Nacional Madidi y Rurrenabaque | La Paz | 6 a 8 días | Aventura · Alta · Media · Rústico · Largo · Cálido |
| R04 | Parque Nacional Sajama | Oruro | 2 a 3 días | Aventura · Alta · Baja · Rústico · Corto · Frío |
| R05 | Samaipata y El Fuerte | Santa Cruz | 3 a 4 días | Cultura · Moderada · Media · Tradicional · Corto · Templado |
| R06 | Los Yungas y Coroico | La Paz | 3 a 5 días | Relajación · Moderada · Baja · Resort · Mediano · Cálido |
| R07 | Parque Nacional Toro Toro | Potosí | 2 a 3 días | Aventura · Alta · Media · Rústico · Corto · Cálido |
| R08 | Sucre, la ciudad blanca | Chuquisaca | 3 a 4 días | Cultura · Baja · Media · Moderno · Corto · Templado |
| R09 | Misiones Jesuíticas de Chiquitos | Santa Cruz | 5 a 7 días | Cultura · Baja · Alta · Tradicional · Mediano · Cálido |
| R10 | Valle de la Luna y La Paz | La Paz | 2 a 3 días | Cultura · Baja · Media · Moderno · Corto · Frío |
| R11 | Tarija y la Ruta del Vino | Tarija | 3 a 4 días | Relajación · Baja · Alta · Tradicional · Corto · Templado |
| R12 | Cochabamba e Incallajta | Cochabamba | 4 a 6 días | Cultura · Moderada · Alta · Tradicional · Mediano · Templado |
| R13 | Parque Nacional Noel Kempff Mercado | Santa Cruz | 7 a 10 días | Aventura · Alta · Baja · Rústico · Largo · Cálido |
| R14 | Potosí, la Ciudad Imperial | Potosí | 2 a 3 días | Cultura · Baja · Media · Tradicional · Corto · Frío |
| R15 | Carnaval de Oruro | Oruro | 2 a 4 días | Cultura · Moderada · Alta · Tradicional · Corto · Frío |
| R16 | Parque Nacional Amboró | Santa Cruz | 4 a 6 días | Aventura · Alta · Media · Rústico · Mediano · Cálido |
| R17 | Pampas del Yacuma | Beni | 3 a 5 días | Aventura · Moderada · Media · Rústico · Mediano · Cálido |
| R18 | Tarabuco y su Feria Artesanal | Chuquisaca | 1 a 2 días | Cultura · Baja · Alta · Tradicional · Corto · Templado |
| R19 | Tupiza y la Ruta del Che | Potosí | 3 a 5 días | Aventura · Moderada · Baja · Tradicional · Mediano · Templado |
| R20 | Laguna Colorada y las Lagunas del Altiplano | Potosí | 3 a 4 días | Aventura · Baja · Baja · Rústico · Mediano · Frío |
| R21 | Bermejo | Tarija | 1 a 2 días | Aventura · Baja · Media · Tradicional · Corto · Cálido |
| R22 | Aguas Calientes | Santa Cruz | 1 a 2 días | Relajación · Baja · Media · Resort · Corto · Cálido |
| R23 | Chochís | Santa Cruz | 2 a 3 días | Aventura · Alta · Baja · Rústico · Corto · Cálido |
| R24 | Copacabana | La Paz | 2 a 3 días | Cultura · Baja · Alta · Tradicional · Corto · Frío |
| R25 | Cotagaita | Potosí | 1 a 2 días | Cultura · Baja · Media · Tradicional · Corto · Templado |
| R26 | Espejillos | Santa Cruz | 1 día | Aventura · Moderada · Media · Rústico · Corto · Cálido |
| R27 | Padcaya | Tarija | 2 días | Relajación · Baja · Media · Tradicional · Corto · Templado |
| R28 | Oruro | Oruro | 2 a 3 días | Cultura · Baja · Alta · Moderno · Corto · Frío |
| R29 | Roboré | Santa Cruz | 2 a 3 días | Aventura · Moderada · Baja · Rústico · Mediano · Cálido |
| R30 | Rurrenabaque | Beni | 3 a 5 días | Aventura · Alta · Media · Tradicional · Mediano · Cálido |
| R31 | San Ignacio de Moxos | Beni | 2 a 3 días | Cultura · Moderada · Alta · Tradicional · Mediano · Cálido |
| R32 | Santa Cruz de la Sierra | Santa Cruz | 3 a 4 días | Cultura · Baja · Alta · Moderno · Corto · Cálido |
| R33 | Santiago de Chiquitos | Santa Cruz | 2 a 3 días | Cultura · Moderada · Baja · Tradicional · Corto · Cálido |
| R34 | Tiahuanaco | La Paz | 1 día | Cultura · Moderada · Baja · Tradicional · Corto · Frío |
| R35 | Vallegrande | Santa Cruz | 2 días | Cultura · Moderada · Baja · Tradicional · Corto · Templado |
| R36 | Villazón | Potosí | 1 a 2 días | Aventura · Baja · Media · Tradicional · Corto · Frío |
| R37 | Yacuiba | Tarija | 1 a 2 días | Cultura · Baja · Media · Tradicional · Corto · Cálido |

Cada tarjeta muestra un único enlace externo, **Más información del destino**,
que apunta a la guía adaptada de
[BoliviaTurismo.com.bo](https://www.boliviaturismo.com.bo/) (por ejemplo,
`https://www.boliviaturismo.com.bo/salar-de-uyuni.html` para R01). La base de
conocimiento muestra ese mismo enlace por regla.

## Ampliar el sistema

- **Nuevo destino:** agrega un objeto `Regla(...)` a `BASE_CONOCIMIENTO` con sus
  datos (departamento, descripción, duración ideal, altitud/clima, actividades,
  `url_turismo` de BoliviaTurismo y sus 6 condiciones).
  La red bayesiana se reconstruye automáticamente al importar `app.py`.
- **Nueva variable:** añádela a `VARIABLES` (el formulario se genera solo) y
  úsala en las condiciones de las reglas; la CPT se genera sola.
- **Cambiar el número de recomendaciones:** modifica `TOP_RECOMENDACIONES`.
- **Ajustar la fuerza de la evidencia:** modifica `PROB_COINCIDENCIA` (0,75 por
  defecto). Valores más altos hacen que la red confíe más en coincidencias
  exactas; más bajos, que el prior tenga más peso.
