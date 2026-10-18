import pandas as pd
import numpy as np


# PUNTO 1: Cargar las particiones asignadas de manera personalizada ------------------------------------------------

# 'index_col="id"' toma la columna id como índice y ya no la cuenta como columna de atributos
df_E = pd.read_csv("Tarea05_Luis_E.csv", index_col="id")
df_P = pd.read_csv("Tarea05_Luis_P.csv", index_col="id")


# Guardas los IDs directamente desde el índice del DataFrame de prueba
ids_prueba = df_P.index.tolist()
clase_objetivo = 'Class'


# Se extraen los dominios completos de los atributos categóricos en E para gestionar ramas sin patrones

dominios_E = {
    col: sorted(df_E[col].unique())
    for col in df_E.columns
    if col != clase_objetivo and not pd.api.types.is_numeric_dtype(df_E[col])
}


# ------------------------------------------------------------------------------------------------------
# Medidas de impureza (desorden)

def calcular_entropia(col):
    """Calcula la entropía H(E) según la fórmula de Shannon."""
    total = len(col)
    if total == 0:
        return 0.0
    conteos = col.value_counts()
    entropia = 0.0
    for c in conteos:
        if c > 0:
            p = c / total
            entropia -= p * np.log2(p)
    return entropia

def calcular_gini(col):
    """Calcula el índice de impureza de Simpson-Gini: 1 - sum(pi^2)."""
    total = len(col)
    if total == 0:
        return 0.0
    conteos = col.value_counts()
    return 1.0 - sum((c / total) ** 2 for c in conteos)

def calcular_error(col):
    """Calcula el error de clasificación: 1 - max(pi)."""
    total = len(col)
    if total == 0:
        return 0.0
    conteos = col.value_counts()
    return 1.0 - (conteos.max() / total)

def calcular_minimo(col, num_clases=2):
    """Calcula la mínima probabilidad: min(pi)."""
    total = len(col)
    if total == 0:
        return 0.0
    conteos = col.value_counts()
    # Si alguna clase presenta 0 patrones en el nodo, la mínima probabilidad es 0
    if len(conteos) < num_clases:
        return 0.0
    return conteos.min() / total

def obtener_clase_mayoritaria(df, clase_objetivo='Class'):
    """Determina la clase mayoritaria y da formato a la etiqueta de la hoja '+ (aciertos/errores)'."""
    conteos = df[clase_objetivo].value_counts()
    mayoritaria = conteos.index[0]
    total = len(df)
    aciertos = conteos.iloc[0]
    errores = total - aciertos
    if errores > 0:
        etiqueta = f"{mayoritaria} ({aciertos}/{errores})"
    else:
        etiqueta = f"{mayoritaria} ({aciertos})"
    return mayoritaria, etiqueta

# ------------------------------------------------------------------------------------------------------
# Evaluación de Atributos (Categóricos y Numéricos)
# -----------------------------------------------------------------------------
def evaluar_categorico(df, atributo, clase_objetivo='Class'):
    """Evalúa un atributo categórico generando una rama por valor y verificando el mínimo de patrones."""
    total = len(df)
    conteos = df[atributo].value_counts()
    
    # Se verifica que al menos 2 ramas contengan un mínimo de 2 patrones
    ramas_con_minimo_2 = sum(1 for c in conteos if c >= 2)
    if ramas_con_minimo_2 < 2:
        return None
    
    entropia_inicial = calcular_entropia(df[clase_objetivo])
    entropia_pond = 0.0
    split_info = 0.0
    
    for valor, conteo in conteos.items():
        sub = df[df[atributo] == valor]
        p = conteo / total
        entropia_pond += p * calcular_entropia(sub[clase_objetivo])
        split_info -= p * np.log2(p)
        
    ganancia = entropia_inicial - entropia_pond
    gain_ratio = ganancia / split_info if split_info != 0 else 0.0
    
    return ganancia, split_info, gain_ratio

def evaluar_numerico(df, atributo, clase_objetivo='Class'):
    """Evalúa un atributo numérico ordenando los valores y buscando el umbral t con mayor ganancia."""
    valores = np.sort(df[atributo].unique())
    if len(valores) < 2:
        return None
        
    total = len(df)
    entropia_inicial = calcular_entropia(df[clase_objetivo])
    
    mejor_t = None
    mejor_ganancia = -1.0
    mejor_split_info = 0.0
    mejor_gain_ratio = -1.0
    
    for i in range(len(valores) - 1):
        t = (valores[i] + valores[i + 1]) / 2.0
        sub_izq = df[df[atributo] <= t]
        sub_der = df[df[atributo] > t]
        
        # Se valida que cada rama resultante contenga al menos 2 patrones
        if len(sub_izq) >= 2 and len(sub_der) >= 2:
            p_izq = len(sub_izq) / total
            p_der = len(sub_der) / total
            
            ent_pond = (p_izq * calcular_entropia(sub_izq[clase_objetivo]) + 
                        p_der * calcular_entropia(sub_der[clase_objetivo]))
            ganancia = entropia_inicial - ent_pond
            
            split_info = -(p_izq * np.log2(p_izq) + p_der * np.log2(p_der))
            gain_ratio = ganancia / split_info if split_info != 0 else 0.0
            
            # Se selecciona el umbral candidato t que maximiza la ganancia de información
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_t = t
                mejor_split_info = split_info
                mejor_gain_ratio = gain_ratio
                
    if mejor_t is None: 
        return None
        
    return mejor_t, mejor_ganancia, mejor_split_info, mejor_gain_ratio

# -----------------------------------------------------------------------------
# PUNTO 2: Entrenar el árbol de decisión C4.5 usando los 15 atributos
# ----------------------------------------------------------------------------
def entrenar_C4_5(df, atributos=None, clase_objetivo='Class'):
    """Construir de manera recursiva el árbol de decisión C4.5."""
    if atributos is None:
        atributos = [c for c in df.columns if c != clase_objetivo]
    else:
        atributos = list(atributos)
        
    clase_pred, etiqueta_hoja = obtener_clase_mayoritaria(df, clase_objetivo)
    
    # Condiciones de parada: entropía igual a 0, menos de 4 patrones o sin atributos disponibles
    if calcular_entropia(df[clase_objetivo]) == 0 or len(df) < 4 or len(atributos) == 0:
        return etiqueta_hoja

    # Evaluación iterativa de cada atributo disponible
    candidatos = []
    for atrib in atributos:
        if pd.api.types.is_numeric_dtype(df[atrib]):
            res = evaluar_numerico(df, atrib, clase_objetivo)
            if res is not None and res[1] > 0:
                t, gan, s_info, gr = res
                candidatos.append({
                    'atributo': atrib,
                    'es_numerico': True,
                    'umbral': t,
                    'ganancia': gan,
                    'split_info': s_info,
                    'gain_ratio': gr
                })
        else:
            res = evaluar_categorico(df, atrib, clase_objetivo)
            if res is not None and res[0] > 0:
                gan, s_info, gr = res
                candidatos.append({
                    'atributo': atrib,
                    'es_numerico': False,
                    'umbral': None,
                    'ganancia': gan,
                    'split_info': s_info,
                    'gain_ratio': gr
                })

    # Condición de parada: ningún atributo produce una división útil
    if not candidatos:
        return etiqueta_hoja

    # Heurística de Quinlan: filtrar atributos con Ganancia >= Promedio y elegir el de mayor Gain Ratio
    ganancia_promedio = np.mean([c['ganancia'] for c in candidatos])
    validos = [c for c in candidatos if c['ganancia'] >= ganancia_promedio]
    if not validos:             
        validos = candidatos

    ganador = max(validos, key=lambda c: c['gain_ratio'])
    mejor_atrib = ganador['atributo']

    # Partición del nodo y llamadas recursivas según el tipo de atributo ganador
    arbol = {}
    if ganador['es_numerico']:
        # Los atributos numéricos se mantienen disponibles para reutilizarse en niveles inferiores
        t = ganador['umbral']
        sub_izq = df[df[mejor_atrib] <= t]
        sub_der = df[df[mejor_atrib] > t]
        
        arbol[f"{mejor_atrib} <= {t:.4g}"] = entrenar_C4_5(sub_izq, atributos, clase_objetivo)
        arbol[f"{mejor_atrib} > {t:.4g}"] = entrenar_C4_5(sub_der, atributos, clase_objetivo)
    else:
        # Los atributos categóricos se descartan de la lista para las ramas descendientes
        nuevos_atributos = [a for a in atributos if a != mejor_atrib]
        for valor in dominios_E[mejor_atrib]:
            sub_df = df[df[mejor_atrib] == valor]
            if len(sub_df) == 0:
                # Si el valor no presenta patrones en la submatriz, se asigna una hoja con la clase mayoritaria del padre
                arbol[f"{mejor_atrib} = {valor}"] = f"{clase_pred} (0)"
            else:
                arbol[f"{mejor_atrib} = {valor}"] = entrenar_C4_5(sub_df, nuevos_atributos, clase_objetivo)
            
    return arbol

# Funciones auxiliares para el PUNTO 5: Contabilizar hojas, profundidad y nodos del árbol
def contar_hojas(nodo):
    """Cuenta recursivamente el número total de hojas en el árbol."""
    if not isinstance(nodo, dict):
        return 1
    return sum(contar_hojas(sub) for sub in nodo.values())

def calcular_profundidad(nodo):
    """Calcula la profundidad máxima alcanzada por el árbol de decisión."""
    if not isinstance(nodo, dict):
        return 0
    return 1 + max(calcular_profundidad(sub) for sub in nodo.values())

def contar_nodos_totales(nodo):
    """Cuenta el número total de nodos (internos y hojas) que conforman el árbol."""
    if not isinstance(nodo, dict):
        return 1
    return 1 + sum(contar_nodos_totales(sub) for sub in nodo.values())

def imprimir_arbol(nodo, sangria=0, archivo=None):
    """Imprime la estructura jerárquica del árbol utilizando barras verticales de indentación."""
    prefijo = "|  " * sangria
    if not isinstance(nodo, dict):
        print(f": {nodo}", file=archivo)
        return
    for condicion, sub_arbol in nodo.items():
        if isinstance(sub_arbol, dict):
            print(f"{prefijo}{condicion}", file=archivo)
            imprimir_arbol(sub_arbol, sangria + 1, archivo=archivo)
        else:
            print(f"{prefijo}{condicion}: {sub_arbol}", file=archivo)

# Ejecución del entrenamiento del árbol C4.5
arbol = entrenar_C4_5(df_E, clase_objetivo=clase_objetivo)

# -----------------------------------------------------------------------------
# PUNTOS 3 y 4: Análisis en el Nodo Raíz (Conjunto E con los 15 atributos)
# -----------------------------------------------------------------------------
# Extraer la lista de atributos ignorando la columna objetivo
atributos = [c for c in df_E.columns if c != clase_objetivo]
total_E = len(df_E)

# Impurezas iniciales en la raíz antes de cualquier división (línea base)
H_E = calcular_entropia(df_E[clase_objetivo])
G_E = calcular_gini(df_E[clase_objetivo])
Err_E = calcular_error(df_E[clase_objetivo])
Min_E = calcular_minimo(df_E[clase_objetivo])

resultados_p3 = []
resultados_p4 = []

# Evaluar cada uno de los 15 atributos disponibles
for at in atributos:
    es_num = pd.api.types.is_numeric_dtype(df_E[at])
    # --- RAMA 1: ATRIBUTOS NUMÉRICOS (Búsqueda de umbral continuo) ---
    if es_num:
        # Ordenar valores únicos para probar puntos medios como posibles umbrales
        valores = np.sort(df_E[at].unique())
        mejor_t = None
        max_ganancia = -1.0
        mejor_split_info = 0.0
        mejor_gr = -1.0
        mejor_red_H = mejor_red_G = mejor_red_Err = mejor_red_Min = 0.0    
        # Probar cada punto medio consecutivo
        for i in range(len(valores) - 1):
            t = (valores[i] + valores[i + 1]) / 2.0
            sub_izq = df_E[df_E[at] <= t]
            sub_der = df_E[df_E[at] > t]  
            # Condición de validez: al menos 2 patrones por rama
            if len(sub_izq) >= 2 and len(sub_der) >= 2:
                p_izq = len(sub_izq) / total_E
                p_der = len(sub_der) / total_E      
                # Entropía ponderada, Ganancia y SplitInfo para este corte
                h_pond = p_izq * calcular_entropia(sub_izq[clase_objetivo]) + p_der * calcular_entropia(sub_der[clase_objetivo])
                gan = H_E - h_pond
                sp_info = -(p_izq * np.log2(p_izq) + p_der * np.log2(p_der))
                gr = gan / sp_info if sp_info != 0 else 0.0        
                # El mejor umbral es el que maximiza la Ganancia de InformaciónD
                if gan > max_ganancia:
                    max_ganancia = gan
                    mejor_t = t
                    mejor_split_info = sp_info
                    mejor_gr = gr
                    # Con el mejor corte identificado, calcular las reducciones de las 4 impurezas (Punto 4)
                    g_pond = p_izq * calcular_gini(sub_izq[clase_objetivo]) + p_der * calcular_gini(sub_der[clase_objetivo])
                    err_pond = p_izq * calcular_error(sub_izq[clase_objetivo]) + p_der * calcular_error(sub_der[clase_objetivo])
                    min_pond = p_izq * calcular_minimo(sub_izq[clase_objetivo]) + p_der * calcular_minimo(sub_der[clase_objetivo])
                    
                    mejor_red_H = gan
                    mejor_red_G = G_E - g_pond
                    mejor_red_Err = Err_E - err_pond
                    mejor_red_Min = Min_E - min_pond
        
        # Guardar métricas del Punto 3 (C4.5) y Punto 4 (Medidas de impureza)
        resultados_p3.append({
            'Atributo': at, 'Tipo': 'Num.', 'Mejor t': round(mejor_t, 4) 
            if mejor_t is not None else '-', 'Ganancia': max_ganancia, 'SplitInfo': mejor_split_info, 'GainRatio': mejor_gr
        })
        resultados_p4.append({
            'Atributo': at, 'Δ Entropía': mejor_red_H, 'Δ Gini': mejor_red_G, 'Δ Error': mejor_red_Err, 'Δ Mínimo': mejor_red_Min
        })
        
    # --- RAMA 2: ATRIBUTOS CATEGÓRICOS (Partición discreta multi-rama) ---
    else:
        conteos = df_E[at].value_counts()
        h_pond = g_pond = err_pond = min_pond = sp_info = 0.0
        
        # Ponderar impurezas y calcular SplitInfo para cada valor distinto
        for val, cnt in conteos.items():
            sub = df_E[df_E[at] == val]
            p = cnt / total_E
            h_pond += p * calcular_entropia(sub[clase_objetivo])
            g_pond += p * calcular_gini(sub[clase_objetivo])
            err_pond += p * calcular_error(sub[clase_objetivo])
            min_pond += p * calcular_minimo(sub[clase_objetivo])
            sp_info -= p * np.log2(p)
            
        gan = H_E - h_pond
        gr = gan / sp_info if sp_info != 0 else 0.0
        
        # Guardar métricas de las variables categóricas
        resultados_p3.append({
            'Atributo': at,
            'Tipo': 'Cat.',
            'Mejor t': '-',
            'Ganancia': gan,
            'SplitInfo': sp_info,
            'GainRatio': gr
        })
        resultados_p4.append({
            'Atributo': at,
            'Δ Entropía': gan,
            'Δ Gini': G_E - g_pond,
            'Δ Error': Err_E - err_pond,
            'Δ Mínimo': Min_E - min_pond
        })

# Generación e impresión de resultados del Punto 3
df_p3 = pd.DataFrame(resultados_p3)
ganancia_promedio = df_p3['Ganancia'].mean()
df_p3['¿≥ promedio?'] = df_p3['Ganancia'].apply(lambda g: 'Sí' if g >= ganancia_promedio else 'No')

print("-" * 85)
print(f"PUNTO 3: TABLA DE EVALUACIÓN EN NODO RAÍZ (Ganancia Promedio = {ganancia_promedio:.4f})")
print("-" * 85)
print(df_p3[['Atributo', 'Tipo', 'Mejor t', 'Ganancia', '¿≥ promedio?', 'SplitInfo', 'GainRatio']].to_string(index=False))

candidatos_validos = df_p3[df_p3['¿≥ promedio?'] == 'Sí']
raiz_elegida = candidatos_validos.loc[candidatos_validos['GainRatio'].idxmax()]
print("\n--> ATRIBUTO ELEGIDO COMO RAÍZ:", raiz_elegida['Atributo'])

# Generación e impresión de resultados del Punto 4
df_p4 = pd.DataFrame(resultados_p4)
print("\n" + "=" * 85)
print("PUNTO 4: REDUCCIÓN DE IMPUREZA EN NODO RAÍZ")
print("=" * 85)
print(df_p4.to_string(index=False))

print("\n--> ATRIBUTO ELEGIDO POR CADA MEDIDA DE IMPUREZA:")
print(f"* Entropía:            {df_p4.loc[df_p4['Δ Entropía'].idxmax(), 'Atributo']}")
print(f"* Simpson-Gini:        {df_p4.loc[df_p4['Δ Gini'].idxmax(), 'Atributo']}")
print(f"* Error Clasificación: {df_p4.loc[df_p4['Δ Error'].idxmax(), 'Atributo']}")
print(f"* Mínima Probabilidad: {df_p4.loc[df_p4['Δ Mínimo'].idxmax(), 'Atributo']}")

# =============================================================================
# PUNTO 5: Reportar número de hojas y profundidad del árbol obtenido
# =============================================================================
num_hojas = contar_hojas(arbol)
profundidad = calcular_profundidad(arbol)
num_nodos = contar_nodos_totales(arbol)

print("\n" + "=" * 85)
print("PUNTO 5: DIMENSIONES DEL ÁRBOL C4.5 OBTENIDO")
print("=" * 85)
print(f"* Número de hojas: {num_hojas}")
print(f"* Profundidad:     {profundidad}")
print(f"* Nodos totales:   {num_nodos}")

# -----------------------------------------------------------------------------
# PUNTO 6: Evaluar el conjunto de prueba (P)
# ----------------------------------------------------------------------------
def clasificar_patron(arbol, patron):
    """Recorre el árbol de decisión para asignar una clase al patrón evaluado."""
    ruta = []
    nodo_actual = arbol
    
    # Descender por el árbol mientras el nodo actual sea un diccionario (nodo de decisión)
    while isinstance(nodo_actual, dict):
        # Evaluar cada regla/rama disponible en el nodo actual
        for clave, sub_arbol in nodo_actual.items():
            # Caso 1: Regla para atributo numérico continuo (evaluación por umbral)
            if '<=' in clave or '>' in clave:
                partes = clave.split()
                atrib, operador, umbral = partes[0], partes[1], float(partes[2])
                val_patron = patron[atrib]
                
                # Verificar si el valor del patrón cumple la desigualdad de la rama
                if operador == '<=' and val_patron <= umbral:
                    ruta.append(clave)
                    nodo_actual = sub_arbol
                    break
                elif operador == '>' and val_patron > umbral:
                    ruta.append(clave)
                    nodo_actual = sub_arbol
                    break
                    
            # Caso 2: Regla para atributo categórico (comparación directa de valor)
            elif '=' in clave:
                partes = clave.split(' = ')
                atrib, valor_esperado = partes[0], partes[1]
                val_patron = str(patron[atrib])
                
                # Coincidencia con la categoría de la rama
                if val_patron == valor_esperado:
                    ruta.append(clave)
                    nodo_actual = sub_arbol
                    break
            
    # Al llegar a una hoja, extraer solo la etiqueta de la clase
    clase_asignada = str(nodo_actual).split()[0]
    return clase_asignada, " -> ".join(ruta)
resultados_prueba = []
# Evaluar uno a uno todos los patrones del conjunto de prueba P
for i, (idx, fila) in enumerate(df_P.iterrows()):
    clase_real = fila[clase_objetivo]
    clase_pred, ruta = clasificar_patron(arbol, fila)
    resultado = "ACIERTO" if (clase_pred == clase_real) else "ERROR"
    
    resultados_prueba.append({
        '#': ids_prueba[i], 'Condiciones evaluadas (ruta)': ruta, 'Clase asignada': clase_pred,'Clase real': clase_real, 'Resultado': resultado})

# Convertir a DataFrame y guardar el reporte completo patrón por patrón en CSV
df_evaluacion = pd.DataFrame(resultados_prueba)
df_evaluacion.to_csv("Reporte_Punto_6_Evaluacion_P.csv", index=False)

# Resumen de métricas globales y desglose por cada clase (+ y -)
resumen_clases = []
total_patrones = len(df_evaluacion)
total_aciertos = (df_evaluacion['Resultado'] == 'ACIERTO').sum()
total_errores = (df_evaluacion['Resultado'] == 'ERROR').sum()

for c in sorted(df_P[clase_objetivo].unique()):
    sub_c = df_evaluacion[df_evaluacion['Clase real'] == c]
    n_pat = len(sub_c)
    ac = (sub_c['Resultado'] == 'ACIERTO').sum()
    err = (sub_c['Resultado'] == 'ERROR').sum()
    resumen_clases.append({
        'Clase': str(c),'Patrones': n_pat,'Aciertos': ac,'Errores': err,'Efectividad': f"{(ac / n_pat) * 100:.2f}%"
    })

# Despliegue en consola del resumen final
print("\n" + "=" * 85)
print("PUNTO 6: RESUMEN DE EVALUACIÓN DEL CONJUNTO DE PRUEBA (P)")
print("=" * 85)
for fila in resumen_clases:
    print(f"Clase {fila['Clase']}: {fila['Patrones']} patrones | Aciertos: {fila['Aciertos']} | Errores: {fila['Errores']} ({fila['Efectividad']})")
# =============================================================================
# PUNTO 7: Medida de desempeño Accuracy
# =============================================================================
accuracy = total_aciertos / total_patrones
print("\n" + "=" * 85)
print("PUNTO 7: MEDIDA DE DESEMPEÑO (ACCURACY)")
print("=" * 85)
print(f"Aciertos totales: {total_aciertos} / {total_patrones}")
print(f"Accuracy:         {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("=" * 85)

# Exportación de todos los resultados al archivo de texto de reporte general
nombre_archivo_reporte = "reporte_C45_resultados.txt"
with open(nombre_archivo_reporte, "w", encoding="utf-8") as f:
    f.write("PUNTO 3: TABLA DE EVALUACIÓN EN NODO RAÍZ\n")
    f.write(f"Ganancia Promedio = {ganancia_promedio:.4f}\n\n")
    f.write(df_p3[['Atributo', 'Tipo', 'Mejor t', 'Ganancia', '¿≥ promedio?', 'SplitInfo', 'GainRatio']].to_string(index=False))
    f.write(f"\n\n--> ATRIBUTO ELEGIDO COMO RAÍZ: {raiz_elegida['Atributo']}\n\n")
    
    f.write("-" * 90 + "\nPUNTO 4: REDUCCIÓN DE IMPUREZA EN NODO RAÍZ\n\n")
    f.write(df_p4.to_string(index=False))
    f.write("\n\n--> ATRIBUTO ELEGIDO POR CADA MEDIDA:\n")
    f.write(f"* Entropía:            {df_p4.loc[df_p4['Δ Entropía'].idxmax(), 'Atributo']}\n")
    f.write(f"* Simpson-Gini:        {df_p4.loc[df_p4['Δ Gini'].idxmax(), 'Atributo']}\n")
    f.write(f"* Error Clasificación: {df_p4.loc[df_p4['Δ Error'].idxmax(), 'Atributo']}\n")
    f.write(f"* Mínima Probabilidad: {df_p4.loc[df_p4['Δ Mínimo'].idxmax(), 'Atributo']}\n\n")
    
    f.write("-" * 90 + "\nPUNTO 5: NÚMERO DE HOJAS, PROFUNDIDAD Y ÁRBOL C4.5\n\n")
    f.write(f"Número de hojas: {num_hojas}\n")
    f.write(f"Profundidad:     {profundidad}\n")
    f.write(f"Nodos totales:   {num_nodos}\n\n")
    imprimir_arbol(arbol, sangria=0, archivo=f)
    
    f.write("\n" + "-" * 90 + "\nPUNTO 6 Y 7: RESUMEN DE PRUEBA Y ACCURACY\n\n")
    for fila in resumen_clases:
        f.write(f"Clase {fila['Clase']}: Patrones={fila['Patrones']}, Aciertos={fila['Aciertos']}, Errores={fila['Errores']}\n")
    f.write(f"\nTotal Patrones en P: {total_patrones}\n")
    f.write(f"Total Aciertos:      {total_aciertos}\n")
    f.write(f"Total Errores:       {total_errores}\n")
    f.write(f"Accuracy:            {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
    f.write("--- DETALLE DE CLASIFICACIÓN DE CADA PATRÓN DE PRUEBA (P) ---\n")
    f.write(df_evaluacion[['#', 'Clase real', 'Clase asignada', 'Resultado']].to_string(index=False))


print(f"\n[OK] Reporte completo generado y almacenado en '{nombre_archivo_reporte}'.")