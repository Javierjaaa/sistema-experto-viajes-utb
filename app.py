"""Sistema Experto para la recomendación de experiencias de viaje en Bolivia.

Proyecto académico de Inteligencia Artificial.

Componentes:
    - Hechos: preferencias del usuario capturadas en el formulario web.
    - Base de conocimiento: 37 destinos emblemáticos de Bolivia modelados como
      reglas de producción IF-THEN (clase `Regla`).
    - Motor de inferencia: red bayesiana ingenua (nodo oculto `Destino` con 37
      estados y 6 nodos observados); calcula P(destino | preferencias) mediante
      inferencia exacta en log-espacio y devuelve las Top N reglas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from flask import Flask, render_template, request

app = Flask(__name__)

TOP_RECOMENDACIONES = 2

# Probabilidad de que un destino "explique" el valor canónico de una variable;
# el resto se reparte por igual entre las opciones restantes de esa variable.
PROB_COINCIDENCIA = 0.75
PROB_MINIMA = 1e-12


# ---------------------------------------------------------------------------
# 1. Universo de discurso: variables (hechos) y sus valores posibles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Variable:
    """Describe un hecho y las etiquetas visibles de sus valores posibles."""

    etiqueta: str
    opciones: tuple[tuple[str, str], ...]

    @property
    def valores(self) -> tuple[str, ...]:
        """Valores canónicos usados por el motor de inferencia."""
        return tuple(valor for valor, _ in self.opciones)


VARIABLES: dict[str, Variable] = {
    "motivacion": Variable(
        "Motivación principal",
        (
            ("Aventura", "Aventura - Naturaleza y adrenalina"),
            ("Relajación", "Relajación - Descanso y bienestar"),
            ("Cultura", "Cultura - Historia y tradiciones"),
        ),
    ),
    "actividad": Variable(
        "Nivel de actividad física",
        (
            ("Baja", "Baja - Ritmo tranquilo"),
            ("Moderada", "Moderada - Caminatas y paseos"),
            ("Alta", "Alta - Deportes y desafíos físicos"),
        ),
    ),
    "social": Variable(
        "Interacción social",
        (
            ("Baja", "Bajo - Prefiero estar tranquilo/a"),
            ("Media", "Medio - Algunas actividades grupales"),
            ("Alta", "Alto - Conocer gente nueva"),
        ),
    ),
    "alojamiento": Variable(
        "Tipo de alojamiento",
        (
            ("Tradicional", "Tradicional - Posadas y hoteles locales"),
            ("Moderno", "Moderno - Hoteles boutique y diseño"),
            ("Rústico", "Rústico - Refugios y cabañas"),
            ("Resort", "Resort - Todo incluido"),
        ),
    ),
    "duracion": Variable(
        "Duración del viaje",
        (
            ("Corto", "Corto - 1 a 3 días"),
            ("Mediano", "Mediano - 4 a 7 días (semana)"),
            ("Largo", "Largo - Más de una semana"),
        ),
    ),
    "clima": Variable(
        "Clima preferido",
        (
            ("Cálido", "Cálido - Altas temperaturas"),
            ("Templado", "Templado - Clima suave"),
            ("Frío", "Frío - Temperaturas bajas"),
        ),
    ),
}


# ---------------------------------------------------------------------------
# 2. Base de conocimiento: destinos de Bolivia
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Regla:
    """Regla IF <condiciones> THEN <destino turístico de Bolivia>."""

    id: str
    nombre: str
    departamento: str
    descripcion: str
    duracion_ideal: str
    altitud: str
    actividades: tuple[str, ...]
    url_turismo: str
    condiciones: dict[str, str]


BASE_CONOCIMIENTO: tuple[Regla, ...] = (
    Regla(
        id="R01",
        nombre="Salar de Uyuni",
        departamento="Potosí",
        descripcion=(
            "El desierto de sal más grande del mundo: horizontes blancos infinitos, "
            "islas de cactus, lagunas de colores y cielos estrellados. Un paisaje "
            "único que se recorre en 4x4 y que cambia por completo en época de lluvias."
        ),
        duracion_ideal="3 a 5 días",
        altitud="3 650 m s. n. m. · clima frío y seco",
        actividades=("Tours 4x4 por el salar", "Observación astronómica", "Fotografía en las salinas"),
        url_turismo="https://www.boliviaturismo.com.bo/salar-de-uyuni.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Alta",
            "social": "Media",
            "alojamiento": "Rústico",
            "duracion": "Mediano",
            "clima": "Frío",
        },
    ),
    Regla(
        id="R02",
        nombre="Lago Titicaca e Isla del Sol",
        departamento="La Paz",
        descripcion=(
            "El lago navegable más alto del mundo y la cuna de la civilización inca. "
            "Se navega hasta la Isla del Sol para recorrer terrazas agrícolas, "
            "ruinas sagradas y comunidades aymaras que conservan sus tradiciones."
        ),
        duracion_ideal="2 a 3 días",
        altitud="3 812 m s. n. m. · clima frío y seco",
        actividades=("Navegación en lancha", "Sitios arqueológicos incas", "Convivencia con comunidades"),
        url_turismo="https://www.boliviaturismo.com.bo/isla-del-sol.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Moderada",
            "social": "Media",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Frío",
        },
    ),
    Regla(
        id="R03",
        nombre="Parque Nacional Madidi y Rurrenabaque",
        departamento="La Paz",
        descripcion=(
            "Uno de los parques con mayor biodiversidad del planeta, entre la selva "
            "amazónica y los Andes. Desde Rurrenabaque se accede en bote a lodges "
            "selváticos para avistar aves, monos y caimanes con guías locales."
        ),
        duracion_ideal="6 a 8 días",
        altitud="200 a 6 000 m s. n. m. · clima cálido y húmedo",
        actividades=("Senderismo selvático", "Avistamiento de fauna", "Navegación por ríos"),
        url_turismo="https://www.boliviaturismo.com.bo/parque-nacional-madidi.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Alta",
            "social": "Media",
            "alojamiento": "Rústico",
            "duracion": "Largo",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R04",
        nombre="Parque Nacional Sajama",
        departamento="Oruro",
        descripcion=(
            "El primer parque nacional de Bolivia, dominado por el nevado Sajama "
            "(6 542 m). Bosques de queñua, aguas termales y ascensiones de altura "
            "en un entorno de extrema tranquilidad."
        ),
        duracion_ideal="2 a 3 días",
        altitud="4 200 m s. n. m. · clima frío extremo",
        actividades=("Ascensiones y trekking de altura", "Aguas termales", "Fotografía de bosques de queñua"),
        url_turismo="https://www.boliviaturismo.com.bo/oruro.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Alta",
            "social": "Baja",
            "alojamiento": "Rústico",
            "duracion": "Corto",
            "clima": "Frío",
        },
    ),
    Regla(
        id="R05",
        nombre="Samaipata y El Fuerte",
        departamento="Santa Cruz",
        descripcion=(
            "Pueblo de montaña con clima templado y el sitio arqueológico de El Fuerte, "
            "declarado Patrimonio de la Humanidad. Rodeado de bosques nublados, "
            "cascadas y viñedos artesanales."
        ),
        duracion_ideal="3 a 4 días",
        altitud="1 650 m s. n. m. · clima templado",
        actividades=("Visita a El Fuerte", "Caminatas a cascadas", "Turismo rural y viñedos"),
        url_turismo="https://www.boliviaturismo.com.bo/samaipata.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Moderada",
            "social": "Media",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Templado",
        },
    ),
    Regla(
        id="R06",
        nombre="Los Yungas y Coroico",
        departamento="La Paz",
        descripcion=(
            "Descenso desde el altiplano hacia los valles subtropicales por la "
            "histórica Ruta de la Muerte. Café de altura, cascadas, piscinas "
            "naturales y cabañas con vista a la selva para descansar sin prisa."
        ),
        duracion_ideal="3 a 5 días",
        altitud="1 500 m s. n. m. · clima cálido y húmedo",
        actividades=("Cafetales y rutas del café", "Cascadas y piscinas naturales", "Descanso en cabañas"),
        url_turismo="https://www.boliviaturismo.com.bo/coroico.html",
        condiciones={
            "motivacion": "Relajación",
            "actividad": "Moderada",
            "social": "Baja",
            "alojamiento": "Resort",
            "duracion": "Mediano",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R07",
        nombre="Parque Nacional Toro Toro",
        departamento="Potosí",
        descripcion=(
            "Cañones de piedra caliza, cavernas con formaciones fósiles y huellas "
            "de dinosaurios en un paisaje semiárido. Un destino de aventura poco "
            "masificado, ideal para caminar y explorar en grupos pequeños."
        ),
        duracion_ideal="2 a 3 días",
        altitud="2 700 m s. n. m. · clima cálido y seco",
        actividades=("Espeleología y cavernas", "Huellas de dinosaurios", "Trekking por cañones"),
        url_turismo="https://www.boliviaturismo.com.bo/potosi.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Alta",
            "social": "Media",
            "alojamiento": "Rústico",
            "duracion": "Corto",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R08",
        nombre="Sucre, la ciudad blanca",
        departamento="Chuquisaca",
        descripcion=(
            "Capital constitucional de Bolivia y Patrimonio de la Humanidad. Casco "
            "histórico colonial, museos, iglesias y una gastronomía reconocida, "
            "para recorrer a pie sin apuro y con excelente oferta hotelera."
        ),
        duracion_ideal="3 a 4 días",
        altitud="2 800 m s. n. m. · clima templado",
        actividades=("Casco histórico y museos", "Gastronomía y cafés", "Miradores y plazas"),
        url_turismo="https://www.boliviaturismo.com.bo/sucre.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Baja",
            "social": "Media",
            "alojamiento": "Moderno",
            "duracion": "Corto",
            "clima": "Templado",
        },
    ),
    Regla(
        id="R09",
        nombre="Misiones Jesuíticas de Chiquitos",
        departamento="Santa Cruz",
        descripcion=(
            "Circuito de iglesias coloniales construidas por las misiones jesuitas, "
            "Patrimonio de la Humanidad. Se combina con festivales de música barroca, "
            "talleres de artesanos y la vida tranquila de los pueblos chiquitanos."
        ),
        duracion_ideal="5 a 7 días",
        altitud="300 a 500 m s. n. m. · clima cálido tropical",
        actividades=("Circuito de iglesias misionales", "Festivales de música barroca", "Artesanía y gastronomía local"),
        url_turismo="https://www.boliviaturismo.com.bo/san-jose-de-chiquitos.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Baja",
            "social": "Alta",
            "alojamiento": "Tradicional",
            "duracion": "Mediano",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R10",
        nombre="Valle de la Luna y La Paz",
        departamento="La Paz",
        descripcion=(
            "Formaciones de arcilla erosionada a minutos de la ciudad más alta del "
            "mundo. Se completa con teleférico, mercados de brujería y miradores "
            "urbanos: cultura y paisaje en formato corto."
        ),
        duracion_ideal="2 a 3 días",
        altitud="3 600 m s. n. m. · clima frío",
        actividades=("Valle de la Luna", "Paseo en teleférico", "Mercados y miradores urbanos"),
        url_turismo="https://www.boliviaturismo.com.bo/la-paz.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Baja",
            "social": "Media",
            "alojamiento": "Moderno",
            "duracion": "Corto",
            "clima": "Frío",
        },
    ),
    Regla(
        id="R11",
        nombre="Tarija y la Ruta del Vino",
        departamento="Tarija",
        descripcion=(
            "Valles templados de viñedos y bodegas artesanales donde se produce la "
            "mayor parte del vino boliviano. Peñas folclóricas, gastronomía local "
            "y un ritmo relajado para degustar sin itinerarios exigentes."
        ),
        duracion_ideal="3 a 4 días",
        altitud="1 850 m s. n. m. · clima templado",
        actividades=("Visitas a bodegas", "Degustación de vinos y singani", "Peñas y música folclórica"),
        url_turismo="https://www.boliviaturismo.com.bo/tarija.html",
        condiciones={
            "motivacion": "Relajación",
            "actividad": "Baja",
            "social": "Alta",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Templado",
        },
    ),
    Regla(
        id="R12",
        nombre="Cochabamba e Incallajta",
        departamento="Cochabamba",
        descripcion=(
            "El valle de clima agradable de Bolivia combina mercados, gastronomía y "
            "el sitio arqueológico inca de Incallajta. Un recorrido cultural con "
            "caminatas suaves y buena vida social."
        ),
        duracion_ideal="4 a 6 días",
        altitud="2 600 m s. n. m. · clima templado",
        actividades=("Ruinas incas de Incallajta", "Mercados gastronómicos", "Caminatas por el valle"),
        url_turismo="https://www.boliviaturismo.com.bo/cochabamba.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Moderada",
            "social": "Alta",
            "alojamiento": "Tradicional",
            "duracion": "Mediano",
            "clima": "Templado",
        },
    ),
    Regla(
        id="R13",
        nombre="Parque Nacional Noel Kempff Mercado",
        departamento="Santa Cruz",
        descripcion=(
            "Reserva de selva amazónica y cerrado con cataratas como las Arco Iris, "
            "declarada Patrimonio de la Humanidad. Aventura remota de varios días, "
            "con muy poca infraestructura y contacto pleno con la naturaleza."
        ),
        duracion_ideal="7 a 10 días",
        altitud="200 a 750 m s. n. m. · clima cálido tropical",
        actividades=("Cataratas y ríos", "Safari fotográfico", "Campamento en la selva"),
        url_turismo="https://www.boliviaturismo.com.bo/robore.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Alta",
            "social": "Baja",
            "alojamiento": "Rústico",
            "duracion": "Largo",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R14",
        nombre="Potosí, la Ciudad Imperial",
        departamento="Potosí",
        descripcion=(
            "Patrimonio de la Humanidad y antigua capital minera del imperio español. "
            "Su Cerro Rico, la Casa de la Moneda y sus iglesias coloniales narran "
            "la historia boliviana en cada calle."
        ),
        duracion_ideal="2 a 3 días",
        altitud="4 090 m s. n. m. · clima frío y seco",
        actividades=("Cerro Rico y minas", "Casa de la Moneda", "Casco colonial y gastronomía"),
        url_turismo="https://www.boliviaturismo.com.bo/potosi.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Baja",
            "social": "Media",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Frío",
        },
    ),
    Regla(
        id="R15",
        nombre="Carnaval de Oruro",
        departamento="Oruro",
        descripcion=(
            "Patrimonio Inmaterial de la Humanidad: diabladas, morenadas y comparsas "
            "que desbordan la ciudad en una semana de música, trajes y devoción "
            "a la Virgen del Socavón."
        ),
        duracion_ideal="2 a 4 días",
        altitud="3 735 m s. n. m. · clima frío y seco",
        actividades=("Desfiles de comparsas", "Diablada y morenada", "Virgen del Socavón"),
        url_turismo="https://www.boliviaturismo.com.bo/carnaval-oruro-2027.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Moderada",
            "social": "Alta",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Frío",
        },
    ),
    Regla(
        id="R16",
        nombre="Parque Nacional Amboró",
        departamento="Santa Cruz",
        descripcion=(
            "Reserva de biosfera donde convergen la selva amazónica, el bosque "
            "nublado y el Chaco. Cascadas, senderos y una avifauna extraordinaria "
            "a pocas horas de Samaipata."
        ),
        duracion_ideal="4 a 6 días",
        altitud="300 a 3 000 m s. n. m. · clima cálido y húmedo",
        actividades=("Senderismo en la selva", "Observación de aves", "Cascadas y bosque nublado"),
        url_turismo="https://www.boliviaturismo.com.bo/samaipata.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Alta",
            "social": "Media",
            "alojamiento": "Rústico",
            "duracion": "Mediano",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R17",
        nombre="Pampas del Yacuma",
        departamento="Beni",
        descripcion=(
            "Llanos inundables de la Amazonía beniana recorridos en bote entre "
            "jacarandás y palmeras: caimanes, delfines rosados y aves en un entorno "
            "de naturaleza casi intacta."
        ),
        duracion_ideal="3 a 5 días",
        altitud="150 m s. n. m. · clima cálido y húmedo",
        actividades=("Safari en bote", "Delfines rosados y caimanes", "Paseos por la sabana"),
        url_turismo="https://www.boliviaturismo.com.bo/rurrenabaque.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Moderada",
            "social": "Media",
            "alojamiento": "Rústico",
            "duracion": "Mediano",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R18",
        nombre="Tarabuco y su Feria Artesanal",
        departamento="Chuquisaca",
        descripcion=(
            "Pueblo de tradición yamparúa famoso por su feria dominical de tejidos "
            "y artesanías. Una explosión de colores, danzas y gastronomía serrana "
            "a una hora de Sucre."
        ),
        duracion_ideal="1 a 2 días",
        altitud="3 300 m s. n. m. · clima templado y seco",
        actividades=("Feria artesanal dominical", "Danzas y trajes típicos", "Telares y gastronomía local"),
        url_turismo="https://www.boliviaturismo.com.bo/tarabuco.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Baja",
            "social": "Alta",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Templado",
        },
    ),
    Regla(
        id="R19",
        nombre="Tupiza y la Ruta del Che",
        departamento="Potosí",
        descripcion=(
            "Quebradas rojizas, cactus y paisajes de western en los valles del sur "
            "boliviano. Desde aquí se recorren los escenarios de la última guerrilla "
            "del Che Guevara y se cabalga hacia la frontera argentina."
        ),
        duracion_ideal="3 a 5 días",
        altitud="1 700 m s. n. m. · clima templado y seco",
        actividades=("Ruta del Che", "Cabalgatas por quebradas", "Miradores y fotografía"),
        url_turismo="https://www.boliviaturismo.com.bo/tupiza.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Moderada",
            "social": "Baja",
            "alojamiento": "Tradicional",
            "duracion": "Mediano",
            "clima": "Templado",
        },
    ),
    Regla(
        id="R20",
        nombre="Laguna Colorada y las Lagunas del Altiplano",
        departamento="Potosí",
        descripcion=(
            "Espejos de agua rojiza, verde y azul rodeados de flamencos y volcanes "
            "en la reserva Eduardo Avaroa. Un circuito de alta montaña de paisajes "
            "irreales que se recorre en 4x4 desde Uyuni."
        ),
        duracion_ideal="3 a 4 días",
        altitud="4 278 m s. n. m. · clima frío extremo",
        actividades=("Lagunas de colores", "Flamencos y fauna altiplánica", "Fotografía de volcanes"),
        url_turismo="https://www.boliviaturismo.com.bo/salar-de-uyuni.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Baja",
            "social": "Baja",
            "alojamiento": "Rústico",
            "duracion": "Mediano",
            "clima": "Frío",
        },
    ),
    Regla(
        id="R21",
        nombre="Bermejo",
        departamento="Tarija",
        descripcion=(
            "Ciudad fronteriza con Argentina a orillas del río Bermejo, unida a Aguas "
            "Blancas por puente internacional y chalanas. Centro comercial con ferias "
            "y caña de azúcar, puerta de entrada a la Reserva Nacional de Tariquía."
        ),
        duracion_ideal="1 a 2 días",
        altitud="500 m s. n. m. · clima cálido subtropical",
        actividades=("Cruce a Aguas Blancas", "Playas y cañadones del río", "Reserva Nacional de Tariquía"),
        url_turismo="https://www.boliviaturismo.com.bo/bermejo.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Baja",
            "social": "Media",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R22",
        nombre="Aguas Calientes",
        departamento="Santa Cruz",
        descripcion=(
            "Comunidad a 32 km de Roboré, en la carretera Bioceánica, famosa por sus "
            "aguas termales de propiedades medicinales. Sus tres atractivos son "
            "Los Hervores, El Burriño y El Puente."
        ),
        duracion_ideal="1 a 2 días",
        altitud="300 m s. n. m. · clima cálido chiquitano",
        actividades=("Baños en Los Hervores", "Lodo y natación en El Burriño", "Relax en El Puente"),
        url_turismo="https://www.boliviaturismo.com.bo/aguas-calientes.html",
        condiciones={
            "motivacion": "Relajación",
            "actividad": "Baja",
            "social": "Media",
            "alojamiento": "Resort",
            "duracion": "Corto",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R23",
        nombre="Chochís",
        departamento="Santa Cruz",
        descripcion=(
            "Población a 40 km de Roboré en la ruta Bioceánica, ideal para descansar "
            "y disfrutar la naturaleza. Destacan la Serranía de Chochís, la Torre "
            "de David y el Santuario Mariano."
        ),
        duracion_ideal="2 a 3 días",
        altitud="1 200 m s. n. m. · clima cálido-templado",
        actividades=("Caminata al Bosque de Piedra", "Torre de David y Santuario", "Pozas y El Velo de la Novia"),
        url_turismo="https://www.boliviaturismo.com.bo/chochis.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Alta",
            "social": "Baja",
            "alojamiento": "Rústico",
            "duracion": "Corto",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R24",
        nombre="Copacabana",
        departamento="La Paz",
        descripcion=(
            "Centro turístico y de peregrinación a orillas del Lago Titicaca, a 150 km "
            "de La Paz y puerta de entrada a la Isla del Sol. Su Basílica, iniciada "
            "en 1601, guarda a la Virgen, patrona de Bolivia."
        ),
        duracion_ideal="2 a 3 días",
        altitud="3 800 m s. n. m. · clima frío de altiplano",
        actividades=("Basílica y Calvario", "Trekking a Yampupata", "Navegación a la Isla del Sol"),
        url_turismo="https://www.boliviaturismo.com.bo/copacabana.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Baja",
            "social": "Alta",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Frío",
        },
    ),
    Regla(
        id="R25",
        nombre="Cotagaita",
        departamento="Potosí",
        descripcion=(
            "Ciudad tranquila del valle de Nor Chichas, capital frutícola y folklórica "
            "de Potosí. Escenario de las batallas de Cotagaita y Tumusla, con "
            "yacimientos arqueológicos en los alrededores."
        ),
        duracion_ideal="1 a 2 días",
        altitud="2 624 m s. n. m. · clima agradable de valle",
        actividades=("Cementerio prehispánico de Caytola", "Plazuela de Tumusla", "Feria frutícola y festival folklórico"),
        url_turismo="https://www.boliviaturismo.com.bo/cotagaita.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Baja",
            "social": "Media",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Templado",
        },
    ),
    Regla(
        id="R26",
        nombre="Espejillos",
        departamento="Santa Cruz",
        descripcion=(
            "Balneario natural de cascadas y saltos de agua a 45 km de Santa Cruz, "
            "dentro de una reserva natural y puerta de entrada al Parque Nacional "
            "Amboró. Rodeado de vegetación selvática, ideal para baños y caminatas."
        ),
        duracion_ideal="1 día",
        altitud="400 m s. n. m. · clima cálido y húmedo",
        actividades=("Baños bajo las cascadas", "Senderos de la reserva", "Camping y picnic"),
        url_turismo="https://www.boliviaturismo.com.bo/espejillos.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Moderada",
            "social": "Media",
            "alojamiento": "Rústico",
            "duracion": "Corto",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R27",
        nombre="Padcaya",
        departamento="Tarija",
        descripcion=(
            "Pueblo en un valle rodeado de cerros, a 50 km al sur de Tarija, ideal "
            "para descansar entre viñedos y paisajes. Cerca están el Santuario de "
            "Chaguaya y el Valle de los Cóndores."
        ),
        duracion_ideal="2 días",
        altitud="2 041 m s. n. m. · clima agradable de valle",
        actividades=("Santuario de Chaguaya", "Valle de los Cóndores", "Viñedos y plaza principal"),
        url_turismo="https://www.boliviaturismo.com.bo/padcaya.html",
        condiciones={
            "motivacion": "Relajación",
            "actividad": "Baja",
            "social": "Media",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Templado",
        },
    ),
    Regla(
        id="R28",
        nombre="Oruro",
        departamento="Oruro",
        descripcion=(
            "Ciudad cabecera del departamento, famosa por su Carnaval Patrimonio de "
            "la Humanidad y por sus tradiciones mineras y nativas. Santuario de la "
            "Virgen del Socavón, museos, termas de Obrajes y el Lago Poopó."
        ),
        duracion_ideal="2 a 3 días",
        altitud="3 735 m s. n. m. · clima frío y seco",
        actividades=("Virgen del Socavón en teleférico", "Carnaval: diablada y morenada", "Termas de Obrajes y Lago Poopó"),
        url_turismo="https://www.boliviaturismo.com.bo/oruro.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Baja",
            "social": "Alta",
            "alojamiento": "Moderno",
            "duracion": "Corto",
            "clima": "Frío",
        },
    ),
    Regla(
        id="R29",
        nombre="Roboré",
        departamento="Santa Cruz",
        descripcion=(
            "Ciudad chiquitana convertida en base de circuitos naturales, con "
            "cascadas y balnearios de interés nacional e internacional. Cuenta con "
            "hospedaje y terminal de buses y tren."
        ),
        duracion_ideal="2 a 3 días",
        altitud="300 m s. n. m. · clima cálido tropical",
        actividades=("Chorro de San Luis", "Cascada de los Helechos", "Baños en San Manuel y centro artesanal"),
        url_turismo="https://www.boliviaturismo.com.bo/robore.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Moderada",
            "social": "Baja",
            "alojamiento": "Rústico",
            "duracion": "Mediano",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R30",
        nombre="Rurrenabaque",
        departamento="Beni",
        descripcion=(
            "Pequeña ciudad del norte de Bolivia a orillas del río Beni, puerta de "
            "entrada a la selva amazónica. Desde aquí parten las excursiones al "
            "Parque Nacional Madidi y a las Pampas de Yacuma."
        ),
        duracion_ideal="3 a 5 días",
        altitud="200 m s. n. m. · clima cálido y húmedo",
        actividades=("Selva del Parque Nacional Madidi", "Safari en las Pampas de Yacuma", "Navegación por los ríos Beni y Tuichi"),
        url_turismo="https://www.boliviaturismo.com.bo/rurrenabaque.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Alta",
            "social": "Media",
            "alojamiento": "Tradicional",
            "duracion": "Mediano",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R31",
        nombre="San Ignacio de Moxos",
        departamento="Beni",
        descripcion=(
            "Pueblo jesuítico del Beni conocido por su música barroca y la Ichapekene "
            "Piesta, Patrimonio Cultural Inmaterial de la Humanidad. Combina "
            "historia misional con la naturaleza amazónica."
        ),
        duracion_ideal="2 a 3 días",
        altitud="150 m s. n. m. · clima cálido y húmedo",
        actividades=("Templo misional y museo", "Laguna Isireri", "Ichapekene Piesta y gastronomía beniana"),
        url_turismo="https://www.boliviaturismo.com.bo/san-ignacio-de-moxos.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Moderada",
            "social": "Alta",
            "alojamiento": "Tradicional",
            "duracion": "Mediano",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R32",
        nombre="Santa Cruz de la Sierra",
        departamento="Santa Cruz",
        descripcion=(
            "Ciudad más poblada de Bolivia y principal motor económico, en los llanos "
            "del oriente. Su centro colonial en torno a la Plaza 24 de Septiembre es "
            "punto de partida hacia Lomas de Arena y las Misiones Jesuíticas."
        ),
        duracion_ideal="3 a 4 días",
        altitud="400 m s. n. m. · clima cálido y húmedo",
        actividades=("Plaza 24 de Septiembre y Catedral", "Jardín Botánico y zoológico", "Lomas de Arena y Misiones Jesuíticas"),
        url_turismo="https://www.boliviaturismo.com.bo/santa-cruz-de-la-sierra.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Baja",
            "social": "Alta",
            "alojamiento": "Moderno",
            "duracion": "Corto",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R33",
        nombre="Santiago de Chiquitos",
        departamento="Santa Cruz",
        descripcion=(
            "Población misional jesuítica a 22 km de Roboré, con tradiciones vivas "
            "y música misional. Destacan los miradores del Valle de Tucavaca, arcos, "
            "cuevas y pinturas rupestres."
        ),
        duracion_ideal="2 a 3 días",
        altitud="750 m s. n. m. · clima cálido chiquitano",
        actividades=("Miradores del Valle de Tucavaca", "Arco Grande y Cueva de Miserandino", "Iglesia misional y arte sacro"),
        url_turismo="https://www.boliviaturismo.com.bo/santiago-de-chiquitos.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Moderada",
            "social": "Baja",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Cálido",
        },
    ),
    Regla(
        id="R34",
        nombre="Tiahuanaco",
        departamento="La Paz",
        descripcion=(
            "El sitio arqueológico más importante de Bolivia, ruinas de la cultura "
            "preincaica que floreció entre 1500 a.C. y 1200 d.C. Incluye la Puerta "
            "del Sol, la Pirámide de Akapana, Kalasasaya y un museo de sitio."
        ),
        duracion_ideal="1 día",
        altitud="3 885 m s. n. m. · clima frío de altiplano",
        actividades=("Puerta del Sol y Akapana", "Kalasasaya y Monolito de Ponce", "Museo de sitio y Monolito Bennett"),
        url_turismo="https://www.boliviaturismo.com.bo/tiahuanaco.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Moderada",
            "social": "Baja",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Frío",
        },
    ),
    Regla(
        id="R35",
        nombre="Vallegrande",
        departamento="Santa Cruz",
        descripcion=(
            "Localidad tranquila rodeada de sierras y fundada en 1612, con "
            "construcciones coloniales y paisajes de valle. Punto clave de la Ruta "
            "del Che: lavandería del hospital, antigua tumba, mausoleo y museo."
        ),
        duracion_ideal="2 días",
        altitud="2 030 m s. n. m. · clima agradable de valle",
        actividades=("Lavandería y mausoleo del Che", "Museo municipal de la Ruta del Che", "La Higuera y pinturas de Mataral"),
        url_turismo="https://www.boliviaturismo.com.bo/vallegrande.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Moderada",
            "social": "Baja",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Templado",
        },
    ),
    Regla(
        id="R36",
        nombre="Villazón",
        departamento="Potosí",
        descripcion=(
            "Ciudad fronteriza con Argentina y paso más usado por turistas, unida a "
            "La Quiaca por un puente internacional. Localidad tranquila con intenso "
            "movimiento comercial, casas de cambio y terminal de buses y tren."
        ),
        duracion_ideal="1 a 2 días",
        altitud="3 400 m s. n. m. · clima fresco y seco",
        actividades=("Cruce a La Quiaca", "Artesanías y cambio de moneda", "Tren a Uyuni u Oruro y visita a Tupiza"),
        url_turismo="https://www.boliviaturismo.com.bo/villazon.html",
        condiciones={
            "motivacion": "Aventura",
            "actividad": "Baja",
            "social": "Media",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Frío",
        },
    ),
    Regla(
        id="R37",
        nombre="Yacuiba",
        departamento="Tarija",
        descripcion=(
            "Ciudad fronteriza del Chaco boliviano, unida a Salvador Mazza "
            "(Argentina) por el Puente Internacional. Ciudad de paso con intensa "
            "actividad comercial y acceso a pueblos del Chaco y la Aguarague."
        ),
        duracion_ideal="1 a 2 días",
        altitud="400 m s. n. m. · clima cálido del Chaco",
        actividades=("Cruce a Salvador Mazza", "Iglesia San Pedro y mercado central", "Pueblos del Chaco y Serranía de Aguarague"),
        url_turismo="https://www.boliviaturismo.com.bo/yacuiba.html",
        condiciones={
            "motivacion": "Cultura",
            "actividad": "Baja",
            "social": "Media",
            "alojamiento": "Tradicional",
            "duracion": "Corto",
            "clima": "Cálido",
        },
    ),
)


# ---------------------------------------------------------------------------
# 3. Motor de inferencia: red bayesiana
# ---------------------------------------------------------------------------
def evaluar_regla(regla: Regla, hechos: dict[str, str]) -> list[str]:
    """Devuelve las condiciones de la regla que coinciden con los hechos.

    Se muestra en la interfaz como "reglas activadas"; la decisión del motor
    la toma la red bayesiana (véase `RedBayesiana.inferir`).
    """
    coincidencias: list[str] = []
    for variable, valor_esperado in regla.condiciones.items():
        if hechos.get(variable) == valor_esperado:
            etiqueta = VARIABLES[variable].etiqueta
            coincidencias.append(f"{etiqueta}: {valor_esperado}")
    return coincidencias


class RedBayesiana:
    """Red bayesiana ingenua: un nodo oculto `Destino` y un nodo por hecho.

    Estructura:
        Destino (37 estados, prior uniforme) -> Motivación, Actividad,
        Interacción social, Alojamiento, Duración y Clima (nodos observados).

    Para cada destino, P(valor | destino) = PROB_COINCIDENCIA cuando el valor
    es el descrito por su regla; el resto se reparte por igual entre las demás
    opciones de esa variable. La inferencia es exacta y se calcula en
    log-espacio para evitar underflow al multiplicar seis probabilidades.
    """

    def __init__(self, reglas: tuple[Regla, ...]) -> None:
        self.prior = 1 / len(reglas)
        self.cpt: dict[str, dict[str, dict[str, float]]] = {}
        for regla in reglas:
            tabla: dict[str, dict[str, float]] = {}
            for variable, esperado in regla.condiciones.items():
                alternativas = len(VARIABLES[variable].valores) - 1
                prob_otro = (1 - PROB_COINCIDENCIA) / alternativas if alternativas else 0.0
                tabla[variable] = {
                    valor: PROB_COINCIDENCIA if valor == esperado else prob_otro
                    for valor in VARIABLES[variable].valores
                }
            self.cpt[regla.id] = tabla

    def inferir(self, hechos: dict[str, str]) -> dict[str, float]:
        """Devuelve P(destino | hechos) normalizada para las 37 reglas."""
        log_pesos: dict[str, float] = {}
        for destino, tabla in self.cpt.items():
            log_peso = math.log(self.prior)
            for variable, valor in hechos.items():
                probabilidad = tabla.get(variable, {}).get(valor, PROB_MINIMA)
                log_peso += math.log(max(probabilidad, PROB_MINIMA))
            log_pesos[destino] = log_peso

        maximo = max(log_pesos.values())
        pesos = {destino: math.exp(peso - maximo) for destino, peso in log_pesos.items()}
        total = sum(pesos.values())
        return {destino: peso / total for destino, peso in pesos.items()}


RED = RedBayesiana(BASE_CONOCIMIENTO)


def calcular_confianza(porcentaje: float) -> str:
    """Traduce la probabilidad bayesiana (en %) a una etiqueta de confianza."""
    if porcentaje >= 35:
        return "Alta"
    if porcentaje >= 15:
        return "Media"
    return "Baja"


def motor_inferencia(
    hechos: dict[str, str], top_n: int = TOP_RECOMENDACIONES
) -> list[dict[str, object]]:
    """Infiere P(destino | hechos) y devuelve las Top N reglas."""
    posteriores = RED.inferir(hechos)
    evaluaciones: list[dict[str, object]] = []

    for orden, regla in enumerate(BASE_CONOCIMIENTO):
        coincidencias = evaluar_regla(regla, hechos)
        total = len(regla.condiciones)
        probabilidad = round(posteriores[regla.id] * 100, 1)
        evaluaciones.append(
            {
                "orden": orden,
                "id": regla.id,
                "nombre": regla.nombre,
                "departamento": regla.departamento,
                "descripcion": regla.descripcion,
                "duracion_ideal": regla.duracion_ideal,
                "altitud": regla.altitud,
                "actividades": regla.actividades,
                "url_turismo": regla.url_turismo,
                "coincidencias": coincidencias,
                "puntaje": len(coincidencias),
                "total": total,
                "porcentaje": round(len(coincidencias) * 100 / total),
                "probabilidad": probabilidad,
                "confianza": calcular_confianza(probabilidad),
            }
        )

    evaluaciones.sort(key=lambda item: (-item["probabilidad"], item["orden"]))
    mejores = evaluaciones[:top_n]

    # Sin ninguna coincidencia no existe recomendación válida.
    if not mejores or not mejores[0]["puntaje"]:
        return []
    return mejores


# ---------------------------------------------------------------------------
# 4. Controladores (rutas Flask)
# ---------------------------------------------------------------------------
def leer_hechos(formulario) -> tuple[dict[str, str], str | None]:
    """Valida el formulario y construye el diccionario de hechos."""
    hechos: dict[str, str] = {}
    for clave, variable in VARIABLES.items():
        valor = (formulario.get(clave) or "").strip()
        if valor not in variable.valores:
            mensaje = f"Selecciona una opción válida para «{variable.etiqueta}»."
            return {}, mensaje
        hechos[clave] = valor
    return hechos, None


def resumen_base_conocimiento() -> list[dict[str, object]]:
    """Prepara la base de conocimiento para mostrarla en la interfaz."""
    resumen: list[dict[str, object]] = []
    for regla in BASE_CONOCIMIENTO:
        condiciones = [
            {"variable": VARIABLES[var].etiqueta, "valor": valor}
            for var, valor in regla.condiciones.items()
        ]
        resumen.append(
            {
                "id": regla.id,
                "nombre": regla.nombre,
                "departamento": regla.departamento,
                "url_turismo": regla.url_turismo,
                "condiciones": condiciones,
            }
        )
    return resumen


@app.route("/", methods=["GET", "POST"])
def index():
    """Presenta el formulario y, tras el envío, ejecuta el motor de inferencia."""
    hechos: dict[str, str] = {}
    resultados: list[dict[str, object]] = []
    error: str | None = None

    if request.method == "POST":
        hechos, error = leer_hechos(request.form)
        if error is None:
            resultados = motor_inferencia(hechos)

    return render_template(
        "index.html",
        variables=VARIABLES,
        resultados=resultados,
        hechos=hechos,
        error=error,
        evaluado=request.method == "POST" and error is None,
        base_conocimiento=resumen_base_conocimiento(),
    )


if __name__ == "__main__":
    app.run(debug=True)
