# Valuación de una put americana sobre SPY con una PINN

Red neuronal informada por la física (*Physics-Informed Neural Network*, PINN) que valúa una opción put americana sobre SPY (ETF del S&P 500) y aprende su frontera óptima de ejercicio. Los resultados se comparan contra un árbol binomial Cox-Ross-Rubinstein (CRR) como método de referencia.

Proyecto del **Reto Bourbaki & Tec**. El notebook se adaptó de un notebook base del curso sobre el problema de Stefan (frontera libre para la ecuación del calor), cambiando la física a Black-Scholes con condición terminal.

## Resumen de hallazgos

- La PINN reproduce el precio del árbol binomial con buena precisión cerca del dinero. El precio de la put ATM es **18.43 USD** con la PINN contra **17.98 USD** con el árbol, un error de **0.45 USD (2.5 %)**.
- Sobre toda la rejilla (S, t), el error absoluto medio es **1.32 USD**, el RMSE **1.83 USD** y el error relativo medio **0.66 %** (calculado solo donde el precio del árbol supera 0.01 USD).
- El error se concentra donde el precio cambia de forma:
  - **In-the-money:** errores de 7 a 8 USD, entre 4.8 % y 8.7 % del precio.
  - **Out-of-the-money:** error pequeño en dólares (≈0.35 USD) pero grande en términos relativos (21 %). En el extremo, la PINN llega a dar un precio ligeramente negativo (−0.28 USD).
  - La mayor discrepancia puntual es de **10.85 USD**, en S ≈ 422 y t ≈ 0, cerca de la frontera de ejercicio.
- La PINN **no respeta del todo la cota inferior** V ≥ (K − S)⁺: en **1,345 de 4,800 puntos** de la rejilla (28 %) queda más de 0.5 USD por debajo del valor de ejercicio inmediato.
- La frontera óptima de ejercicio del árbol tiene la forma esperada: parte de ≈**501 USD** en t = 0 y sube hasta el strike (**555 USD**) en el vencimiento.
- En una prueba sobre una trayectoria simulada, el **árbol binomial fue más preciso que la PINN** (MAE de 0.69 USD contra 1.78 USD). La PINN no supera al método clásico en exactitud puntual. Su interés está en que es una función continua y diferenciable, sin malla.
- El entrenamiento convergió: la pérdida total bajó de 9.3 a 5.8 × 10⁻⁴ en 20,000 iteraciones (≈7.5 min en GPU). El término de la ecuación diferencial (residuo de Black-Scholes) fue el más difícil de reducir y se estabilizó alrededor de 3 × 10⁻⁴, mientras que las condiciones de frontera bajaron a 10⁻⁵–10⁻⁶.

## Problema

Una put americana puede ejercerse en cualquier momento, así que su valoración es un **problema de frontera libre**: hay que encontrar a la vez el precio V(S, t) y la frontera B(t) por debajo de la cual conviene ejercer. En variables normalizadas (x = S/K, v = V/K, b = B/K) el precio cumple la ecuación de Black-Scholes:

```
v_t + ½ σ² x² v_xx + r x v_x − r v = 0
```

con condición terminal v(x, T) = max(1 − x, 0).

## Método

**Datos y parámetros del contrato** (Fase 0). Se descargan con `yfinance` los cierres diarios de SPY del 2 de enero al 1 de abril de 2025 (60 días de cotización).

| Parámetro | Valor | Origen |
|---|---|---|
| Spot S₀ | 553.05 USD | Cierre del 31-mar-2025 |
| Volatilidad σ | 11.45 % anualizada | Desviación estándar de los rendimientos logarítmicos diarios × √252 |
| Strike K | 555 USD | ATM, redondeado al múltiplo de 5 más cercano |
| Tasa libre de riesgo r | 4.4 % | Fija en el código |
| Vencimiento T | 1 año | Fijo en el código |

**Referencia** (Fase 1). Árbol binomial CRR por inducción hacia atrás. Aporta tres cosas: el precio de referencia, la frontera de ejercicio empírica B(t) y las etiquetas V(S, t) para el término de datos.

**Modelo** (Fases 2 y 3). Dos redes acopladas, como en Song et al. (2024):

- `v_net`: aproxima el precio normalizado v(x, t). Red totalmente conectada de 4 capas ocultas de 100 neuronas, activación `tanh`, inicialización Xavier (30,701 parámetros).
- `b_net`: aproxima la frontera libre b(t) (30,601 parámetros).

Las derivadas v_t, v_x y v_xx se obtienen con diferenciación automática de PyTorch (`autograd`), sin derivar a mano.

**Pérdida compuesta de siete términos** (suma ponderada de errores cuadráticos medios):

| Término | Condición | Peso |
|---|---|---|
| PDE | Residuo de Black-Scholes | 1 |
| Terminal | v(x, T) = max(1 − x, 0) | 10 |
| Frontera (Dirichlet) | v(b(t), t) = 1 − b(t) | 10 |
| Smooth-pasting | v_x(b(t), t) = −1 | 1 |
| Campo lejano | v(x_max, t) = 0 | 1 |
| Frontera en vencimiento | b(T) = 1 | 10 |
| Datos | v ≈ precio binomial / K (400 puntos) | 10 |

**Entrenamiento** (Fase 4). Adam con tasa de aprendizaje inicial de 10⁻³ y decaimiento exponencial (factor 0.9 cada 1,000 pasos), 20,000 iteraciones, mini-batches de 256 puntos muestreados en cada paso, semilla 42.

**Experimento adicional.** Se comparan tres estrategias para generar los puntos de colocación (uniforme, equiespaciada y estirada), entrenando una PINN por estrategia con 8,000 iteraciones, y se grafican el precio en t = 0 y la convergencia de cada una.

## Estructura

```
.
├── SPY_American_Put_PINN.ipynb   # Fases 0 a 5: datos, árbol CRR, PINN, resultados
└── README.md
```

## Cómo ejecutarlo

Requiere Python 3.9 o superior. Se recomienda GPU (el entrenamiento tomó ≈7.5 min con una GPU CUDA).

```bash
pip install numpy matplotlib torch yfinance
jupyter notebook SPY_American_Put_PINN.ipynb
```

Ejecuta el notebook de principio a fin. Si `yfinance` no está instalado, el notebook usa datos sintéticos de respaldo.

## Limitaciones

- El término de datos usa etiquetas del árbol binomial, así que el modelo **no es puramente independiente del método de referencia**: la comparación PINN contra árbol mide qué tan bien la red reproduce al árbol.
- La volatilidad se estima con solo 60 días de historia y la tasa libre de riesgo es un valor fijo.
- La comparación contra "mercado" usa una **trayectoria simulada** del subyacente y **precios de mercado sintéticos** (árbol con volatilidad implícita 12 % mayor más ruido de 0.4 USD). No se usaron cotizaciones reales de opciones.
- Se valúa un solo contrato (un subyacente, un strike, una fecha).

## Referencias

- Wang, S. y Perdikaris, P. (2021), PINNs para problemas de frontera libre (citado en el notebook).
- Song et al. (2024), PINN combinada con dos redes para opciones americanas (citado en el notebook).
- Cox, J., Ross, S. y Rubinstein, M. (1979). *Option pricing: a simplified approach*. Journal of Financial Economics.

## Autor

Emilio Páez de la Mora, A01801224. Tecnológico de Monterrey.
