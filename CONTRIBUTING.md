# Normas de trabajo

Guía breve de cómo trabajamos en el repositorio. El objetivo es que seis
personas puedan avanzar en paralelo sin pisarse y que el historial se
entienda de un vistazo. Si algo no está claro o no encaja, se comenta en el
grupo y se ajusta.

## 1. Antes de empezar

- Prepara el entorno siguiendo el [README](README.md#entorno-de-trabajo)
  (`.venv` + `requirements.txt`).
- Trabaja en los issues que tengas asignados. Si ves algo que falta o un
  error fuera de tu tarea, abre un issue en lugar de arreglarlo por tu cuenta.
- Lee la parte de [`docs/planning.md`](docs/planning.md) que corresponde a
  tu rol.

## 2. Flujo de trabajo

```
issue → rama desde develop → commits → push → pull request a develop → revisión y fusión
```

1. **Actualiza `develop`** antes de empezar:
   ```bash
   git checkout develop
   git pull
   ```
2. **Crea una rama para el issue**, con el formato
   `<tipo>/<n.º de issue>-<descripción-corta>`:
   ```bash
   git checkout -b feature/8-mlp-model
   ```
   El tipo es el mismo que en los commits (`feature`, `fix`, `docs`,
   `chore`, ...). Ejemplos: `feature/6-evaluation-metrics`,
   `fix/15-time-alignment`, `docs/5-data-summary`.
3. **Haz commits pequeños** con el formato de la sección 3.
4. **Sube la rama y abre un pull request hacia `develop`**, nunca hacia
   `main`:
   ```bash
   git push -u origin feature/8-mlp-model
   ```
5. **Avisa a Abel** cuando el PR esté listo. Él lo revisa, pide cambios si
   hace falta, lo fusiona y cierra el issue.

### Ramas principales

- **`develop`**: donde se integra el trabajo de todos durante cada fase.
- **`main`**: versión estable. Al cerrar cada fase se fusiona `develop` en
  `main`, de modo que `main` siempre refleja una fase terminada.

## 3. Commits

### Formato

```
[F<fase>] <tipo>(<ámbito>): <descripción breve>

<cuerpo opcional: qué cambia y por qué, 1–3 líneas>

Refs #<n.º de issue>
```

- **`[F<fase>]`**: fase del proyecto a la que pertenece el cambio, según la
  tabla de fases de `docs/planning.md` (por ejemplo `[F1]` para la muestra
  mínima).
- **`<tipo>`**: qué clase de cambio es (tabla de abajo).
- **`(<ámbito>)`**: opcional; la parte del proyecto afectada, por ejemplo
  `models`, `data`, `evaluation`, `planning`, `readme`.
- **`<descripción>`**: en inglés, en imperativo y corta (`add`, no
  `added`), sin punto final.

### Tipos

| Tipo | Cuándo usarlo |
|---|---|
| `feat` | Funcionalidad nueva (un modelo, una métrica, un script) |
| `fix` | Corrección de un error |
| `docs` | Solo documentación (README, `docs/`, comentarios) |
| `refactor` | Reorganizar código sin cambiar lo que hace |
| `test` | Añadir o corregir pruebas |
| `perf` | Mejora de rendimiento |
| `style` | Formato del código, sin cambios de lógica |
| `build` | Dependencias y entorno (`requirements.txt`) |
| `chore` | Mantenimiento (estructura de carpetas, `.gitignore`) |
| `revert` | Deshacer un commit anterior |

### Ejemplos del propio repositorio

```
[F1] feat(models): add common model interface and column schema
[F1] build: declare joblib dependency
[F1] docs(planning): describe model interface and input feature groups
[F1] chore: ignore data and results contents
```

### Buenas prácticas

- Un cambio lógico por commit: mejor tres commits claros que uno con todo.
- Termina el cuerpo con `Refs #n` para que el commit aparezca enlazado en
  el issue.
- Antes de hacer commit, revisa con `git status` que no se cuela nada que
  no corresponde.

## 4. Pull requests

- **Título**: mismo formato que un commit, resumiendo el PR completo. Por
  ejemplo: `[F1] feat(models): define common model interface`.
- **Descripción**, con cuatro partes:

  ```markdown
  ## Summary
  Qué resuelve el PR, en una o dos frases.

  ## Changes
  - Lista de los cambios principales, por archivo o por módulo.

  ## Verification
  Cómo se comprobó que funciona (qué se ejecutó y qué resultado dio).

  Closes #<n.º de issue>
  ```

- Un PR por issue. Si el trabajo crece mucho, mejor dividir el issue.
- La rama debe estar actualizada con `develop` antes de pedir la revisión.

## 5. Reglas del proyecto

Resumen de las normas de [`docs/planning.md`](docs/planning.md), sección 5:

- **El año 2017 no se toca** hasta la evaluación final: ni para ajustar
  parámetros ni para elegir modelos.
- **Escalas y estadísticos solo con datos de entrenamiento**, nunca con
  validación o prueba.
- **No se suben datos ni resultados.** `data/` y `results/` están
  ignorados en git; los archivos que sí deban compartirse se acordarán
  explícitamente.
- **Los modelos implementan `ModelInterface`** y usan los nombres de
  columnas de `src/schema.py`, sin escribirlos a mano. Ver
  [`src/models/README.md`](src/models/README.md).
- **El código se ejecuta desde la raíz del repositorio**, porque los
  imports son del tipo `from src.models... import ...`.
- **Semillas fijas** (`random_state`) en todo lo que tenga aleatoriedad.
- **Los cambios que afectan a otros se acuerdan antes**: el esquema de
  datos, las funciones comunes o el código de otro miembro.

## 6. Idioma

- **Español**: documentación (`docs/`, README) y comentarios del código.
- **Inglés**: commits, ramas, pull requests, issues y nombres en el código
  (variables, funciones, clases).
