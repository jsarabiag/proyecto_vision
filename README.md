# Proyecto visión por ordenador 
## Sistema de seguridad y Tracking 
### Realizado por Iñaki Juan-Aracil y Javier Sarabia Garciía

Proyecto de Visión por Computador (OpenCV, Python).  
El sistema funciona en dos fases:

1) Login de seguridad: detecta una secuencia de color + forma dentro de un ROI.  
2) Juego (billar): detecta y trackea una bola azul, permite seleccionar agujeros y suma puntos cuando la bola se pierde dentro de un agujero.


## Estructura del proyecto

```text
LAB_PROJECT/
├─ assets/
├─ data/
│  ├─ camera_00.jpg ... camera_08.jpg
│  ├─ camera_calibration_params.npz
│  └─ esquinas/
│     ├─ camera_00_marked.jpg ... camera_08_marked.jpg
├─ src/
│  ├─ calibration.py
│  └─ main.py
├─ template/
│  ├─ appendix/
│  ├─ figures/
│  ├─ frontmatter/
│  └─ mainmatter/
├─ README.md
├─ report.bib
├─ report.tex
└─ tudelft-report.cls
