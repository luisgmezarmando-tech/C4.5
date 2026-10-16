# PUNTO 1: 
import pandas as pd
import numpy as np

df_E = pd.read_csv("Tarea05_Luis_E.csv")
df_P = pd.read_csv("Tarea05_Luis_P.csv")


# -----------------------------------------------------------------------------
# 1. ENTROPÍA Y CLASE MAYORITARIA
# -----------------------------------------------------------------------------
def calcular_entropia(columna_objetivo):
    """Calcula H(E) según Shannon."""
    total = len(columna_objetivo)
    if total == 0:
        return 0.0
    conteos = columna_objetivo.value_counts()
    entropia = 0.0
    for c in conteos:
        p = c / total
        entropia -= p * np.log2(p)
    return entropia

def obtener_clase_mayoritaria(df, clase_objetivo='Class'):
    """Retorna la clase mayoritaria y el desglose de patrones."""
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

# -----------------------------------------------------------------------------
# 2. EVALUACIÓN DE ATRIBUTOS CATEGÓRICOS
# -----------------------------------------------------------------------------
def evaluar_categorico(df, atributo, clase_objetivo='Class'):
    """Evalúa un atributo categórico calculando Ganancia, SplitInfo y GainRatio."""
    total = len(df)
    conteos = df[atributo].value_counts()
    
    # Cada rama resultante debe tener al menos 2 patrones según la regla C4.5 del PDF
    if any(c < 2 for c in conteos):
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

# -----------------------------------------------------------------------------
# 3. EVALUACIÓN DE ATRIBUTOS NUMÉRICOS
# -----------------------------------------------------------------------------
def evaluar_numerico(df, atributo, clase_objetivo='Class'):
    """Busca el umbral candidato t = (v_i + v_{i+1})/2 con mayor ganancia."""
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
        
        # Mínimo 2 patrones en cada rama
        if len(sub_izq) >= 2 and len(sub_der) >= 2:
            p_izq = len(sub_izq) / total
            p_der = len(sub_der) / total
            
            ent_pond = (p_izq * calcular_entropia(sub_izq[clase_objetivo]) + 
                        p_der * calcular_entropia(sub_der[clase_objetivo]))
            ganancia = entropia_inicial - ent_pond
            
            split_info = - (p_izq * np.log2(p_izq) + p_der * np.log2(p_der))
            gain_ratio = ganancia / split_info if split_info != 0 else 0.0
            
            # Se escoge el t de mayor ganancia
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_t = t
                mejor_split_info = split_info
                mejor_gain_ratio = gain_ratio
                
    if mejor_t is None:
        return None
        
    return mejor_t, mejor_ganancia, mejor_split_info, mejor_gain_ratio

# -----------------------------------------------------------------------------
# 4. ENTRENAMIENTO RECURSIVO DEL ÁRBOL
# -----------------------------------------------------------------------------
def entrenar_C4_5(df, atributos=None, clase_objetivo='Class'):
    """Construcción recursiva del árbol C4.5 según las láminas del tema RD 06."""
    if atributos is None:
        atributos = [c for c in df.columns if c != clase_objetivo]
    else:
        atributos = list(atributos)
        
    clase_pred, etiqueta_hoja = obtener_clase_mayoritaria(df, clase_objetivo)
    
    # 1. PARADA POR PUREZA (Entropía = 0)
    if calcular_entropia(df[clase_objetivo]) == 0:
        return etiqueta_hoja
        
    # 2. PARADA POR TAMAÑO MÍNIMO O AGOTAMIENTO DE ATRIBUTOS
    if len(df) < 4 or len(atributos) == 0:
        return etiqueta_hoja

    # 3. EVALUACIÓN DE CANDIDATOS
    candidatos = []
    for atrib in atributos:
        if pd.api.types.is_numeric_dtype(df[atrib]):
            res = evaluar_numerico(df, atrib, clase_objetivo)
            if res is not None:
                t, gan, s_info, gr = res
                candidatos.append({
                    'atributo': atrib,
                    'es_numerico': True,
                    'umbral': t,
                    'ganancia': gan,
                    'gain_ratio': gr
                })
        else:
            res = evaluar_categorico(df, atrib, clase_objetivo)
            if res is not None:
                gan, s_info, gr = res
                candidatos.append({
                    'atributo': atrib,
                    'es_numerico': False,
                    'umbral': None,
                    'ganancia': gan,
                    'gain_ratio': gr
                })

    if not candidatos:
        return etiqueta_hoja

    # 4. FILTRO DE QUINLAN: Ganancia >= Promedio, luego mayor Gain Ratio
    ganancia_promedio = np.mean([c['ganancia'] for c in candidatos])
    validos = [c for c in candidatos if c['ganancia'] >= ganancia_promedio]
    
    if not validos:
        validos = candidatos

    ganador = max(validos, key=lambda c: c['gain_ratio'])
    mejor_atrib = ganador['atributo']

    # 5. RECURSIÓN Y GENERACIÓN DE RAMAS
    arbol = {}
    
    if ganador['es_numerico']:
        t = ganador['umbral']
        sub_izq = df[df[mejor_atrib] <= t]
        sub_der = df[df[mejor_atrib] > t]
        
        arbol[f"{mejor_atrib} <= {t}"] = entrenar_C4_5(sub_izq, atributos, clase_objetivo)
        arbol[f"{mejor_atrib} > {t}"] = entrenar_C4_5(sub_der, atributos, clase_objetivo)
    else:
        nuevos_atributos = [a for a in atributos if a != mejor_atrib]
        for valor in sorted(df[mejor_atrib].unique()):
            sub_df = df[df[mejor_atrib] == valor]
            arbol[f"{mejor_atrib} = {valor}"] = entrenar_C4_5(sub_df, nuevos_atributos, clase_objetivo)
            
    return arbol


def imprimir_arbol(nodo, sangria=0,archivo=None):
    """
    Imprime el árbol de decisión en consola con el formato de barras verticales
   
    """
    prefijo = "|  " * sangria
    
    if not isinstance(nodo, dict):
        print(f": {nodo}",file=archivo)
        return

    for condicion, sub_arbol in nodo.items():
        if isinstance(sub_arbol, dict):
            print(f"{prefijo}{condicion}",file=archivo)
            imprimir_arbol(sub_arbol, sangria + 1, archivo=archivo)
        else:
            print(f"{prefijo}{condicion}: {sub_arbol}",file=archivo)


arbol = entrenar_C4_5(df_E, clase_objetivo='Class')
imprimir_arbol(arbol)

import pandas as pd
import numpy as np

clase_objetivo = 'Class'

# -----------------------------------------------------------------------------
# Medidas de impureza (desorden) según el material del curso
# -----------------------------------------------------------------------------
def calcular_entropia(col):
    """Entropía de Shannon[cite: 18]."""
    total = len(col)
    if total == 0:
        return 0.0
    conteos = col.value_counts()
    return -sum((c / total) * np.log2(c / total) for c in conteos)

def calcular_gini(col):
    """Simpson-Gini: 1 - sum(pi^2)[cite: 18]."""
    total = len(col)
    if total == 0:
        return 0.0
    conteos = col.value_counts()
    return 1.0 - sum((c / total) ** 2 for c in conteos)

def calcular_error(col):
    """Error de clasificación: 1 - max(pi)[cite: 18]."""
    total = len(col)
    if total == 0:
        return 0.0
    conteos = col.value_counts()
    return 1.0 - (conteos.max() / total)

def calcular_minimo(col):
    """Mínima probabilidad: min(pi)[cite: 18]."""
    total = len(col)
    if total == 0:
        return 0.0
    conteos = col.value_counts()
    return conteos.min() / total

# -----------------------------------------------------------------------------
# Puntos 3 y 4: Análisis en el Nodo Raíz (Conjunto E)
# -----------------------------------------------------------------------------
atributos = [c for c in df_E.columns if c != clase_objetivo]
total_E = len(df_E)

# Impurezas iniciales del conjunto de entrenamiento E
H_E = calcular_entropia(df_E[clase_objetivo])
G_E = calcular_gini(df_E[clase_objetivo])
Err_E = calcular_error(df_E[clase_objetivo])
Min_E = calcular_minimo(df_E[clase_objetivo])

resultados_p3 = []
resultados_p4 = []

for at in atributos:
    es_num = pd.api.types.is_numeric_dtype(df_E[at])
    
    if es_num:
        valores = np.sort(df_E[at].unique())
        mejor_t = None
        max_ganancia = -1.0
        mejor_split_info = 0.0
        mejor_gr = -1.0
        
        # Reducciones de impureza para el corte óptimo
        mejor_red_H = 0.0
        mejor_red_G = 0.0
        mejor_red_Err = 0.0
        mejor_red_Min = 0.0
        
        # Búsqueda de puntos medios t = (v_i + v_{i+1}) / 2[cite: 25]
        for i in range(len(valores) - 1):
            t = (valores[i] + valores[i + 1]) / 2.0
            sub_izq = df_E[df_E[at] <= t]
            sub_der = df_E[df_E[at] > t]
            
            # Condición de robustez: al menos 2 patrones por rama[cite: 17, 25]
            if len(sub_izq) >= 2 and len(sub_der) >= 2:
                p_izq = len(sub_izq) / total_E
                p_der = len(sub_der) / total_E
                
                # Entropía ponderada y ganancia
                h_pond = p_izq * calcular_entropia(sub_izq[clase_objetivo]) + p_der * calcular_entropia(sub_der[clase_objetivo])
                gan = H_E - h_pond
                
                # SplitInfo y Razón de Ganancia[cite: 24]
                sp_info = -(p_izq * np.log2(p_izq) + p_der * np.log2(p_der))
                gr = gan / sp_info if sp_info != 0 else 0.0
                
                # Seleccionar el umbral con mayor ganancia[cite: 25]
                if gan > max_ganancia:
                    max_ganancia = gan
                    mejor_t = t
                    mejor_split_info = sp_info
                    mejor_gr = gr
                    
                    # Reducciones de impureza asociadas a este umbral
                    g_pond = p_izq * calcular_gini(sub_izq[clase_objetivo]) + p_der * calcular_gini(sub_der[clase_objetivo])
                    err_pond = p_izq * calcular_error(sub_izq[clase_objetivo]) + p_der * calcular_error(sub_der[clase_objetivo])
                    min_pond = p_izq * calcular_minimo(sub_izq[clase_objetivo]) + p_der * calcular_minimo(sub_der[clase_objetivo])
                    
                    mejor_red_H = gan
                    mejor_red_G = G_E - g_pond
                    mejor_red_Err = Err_E - err_pond
                    mejor_red_Min = Min_E - min_pond
        
        resultados_p3.append({
            'Atributo': at,
            'Tipo': 'Num.',
            'Mejor t': round(mejor_t, 4) if mejor_t is not None else '-',
            'Ganancia': max_ganancia,
            'SplitInfo': mejor_split_info,
            'GainRatio': mejor_gr
        })
        
        resultados_p4.append({
            'Atributo': at,
            'Δ Entropía': mejor_red_H,
            'Δ Gini': mejor_red_G,
            'Δ Error': mejor_red_Err,
            'Δ Mínimo': mejor_red_Min
        })
        
    else:
        # Atributo Categórico: una rama por valor[cite: 11, 26]
        conteos = df_E[at].value_counts()
        h_pond = 0.0
        g_pond = 0.0
        err_pond = 0.0
        min_pond = 0.0
        sp_info = 0.0
        
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

# -----------------------------------------------------------------------------
# Punto 3: Tabla del Nodo Raíz y Selección de Atributo
# -----------------------------------------------------------------------------
df_p3 = pd.DataFrame(resultados_p3)
ganancia_promedio = df_p3['Ganancia'].mean()
df_p3['¿≥ promedio?'] = df_p3['Ganancia'].apply(lambda g: 'Sí' if g >= ganancia_promedio else 'No')

print("=" * 85)
print(f"PUNTO 3: TABLA DE EVALUACIÓN EN NODO RAÍZ (Ganancia Promedio = {ganancia_promedio:.4f})")
print("=" * 85)
print(df_p3[['Atributo', 'Tipo', 'Mejor t', 'Ganancia', '¿≥ promedio?', 'SplitInfo', 'GainRatio']].to_string(index=False))

# Heurística de Quinlan: Ganancia >= Promedio y máximo GainRatio
candidatos_validos = df_p3[df_p3['¿≥ promedio?'] == 'Sí']
raiz_elegida = candidatos_validos.loc[candidatos_validos['GainRatio'].idxmax()]
print("\n--> ATRIBUTO ELEGIDO COMO RAÍZ:", raiz_elegida['Atributo'])

# -----------------------------------------------------------------------------
# Punto 4: Reducción de Impureza por Diferentes Criterios
# -----------------------------------------------------------------------------
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

def clasificar_patron(arbol, patron, clase_defecto='+'):
    """
    Recorre el árbol para clasificar un patrón individual.
    Devuelve la clase asignada y la ruta de condiciones evaluadas.
    """
    ruta = []
    nodo_actual = arbol
    
    while isinstance(nodo_actual, dict):
        condicion_cumplida = False
        
        for clave, sub_arbol in nodo_actual.items():
            # Rama numérica (ej. 'A15 <= 228.0' o 'A15 > 228.0')
            if '<=' in clave or '>' in clave:
                partes = clave.split()
                atrib = partes[0]
                operador = partes[1]
                umbral = float(partes[2])
                val_patron = patron[atrib]
                
                if operador == '<=' and val_patron <= umbral:
                    ruta.append(clave)
                    nodo_actual = sub_arbol
                    condicion_cumplida = True
                    break
                elif operador == '>' and val_patron > umbral:
                    ruta.append(clave)
                    nodo_actual = sub_arbol
                    condicion_cumplida = True
                    break
                    
            # Rama categórica (ej. 'A9 = t')
            elif '=' in clave:
                partes = clave.split(' = ')
                atrib = partes[0]
                valor_esperado = partes[1]
                val_patron = str(patron[atrib])
                
                if val_patron == valor_esperado:
                    ruta.append(clave)
                    nodo_actual = sub_arbol
                    condicion_cumplida = True
                    break
        
        # Si un valor categórico no apareció en el entrenamiento (caso borde)[cite: 17]
        if not condicion_cumplida:
            return clase_defecto, " -> ".join(ruta) + " -> [Sin rama]"
            
    # El nodo hoja suele tener el formato '+ (42)' o '+ (2/1)'
    # Extraemos solo el símbolo de la clase (ej. '+' o '-')
    clase_asignada = str(nodo_actual).split()[0]
    return clase_asignada, " -> ".join(ruta)

# =============================================================================
# PUNTO 6: EVALUACIÓN DEL CONJUNTO DE PRUEBA (P)
# =============================================================================
clase_global_mayoritaria = df_E[clase_objetivo].mode()[0]
resultados_prueba = []

for idx, fila in df_P.iterrows():
    clase_real = fila[clase_objetivo]
    clase_pred, ruta = clasificar_patron(arbol, fila, clase_defecto=clase_global_mayoritaria)
    
    es_acierto = (clase_pred == clase_real)
    resultado = "ACIERTO" if es_acierto else "ERROR"
    
    resultados_prueba.append({
        '#': idx + 1,
        'Condiciones evaluadas (ruta)': ruta,
        'Clase asignada': clase_pred,
        'Clase real': clase_real,
        'Resultado': resultado
    })

df_evaluacion = pd.DataFrame(resultados_prueba)

print("\n" + "=" * 85)
print("PUNTO 6: TABLA DE EVALUACIÓN DEL CONJUNTO DE PRUEBA (P)")
print("=" * 85)
# Mostramos las primeras 10 filas como muestra en consola
print(df_evaluacion[['#', 'Clase asignada', 'Clase real', 'Resultado', 'Condiciones evaluadas (ruta)']].head(15).to_string(index=False))
print(f"\n... Total de patrones evaluados en P: {len(df_evaluacion)}")

# Guardar la tabla completa en un CSV para anexarlo a tu reporte si lo necesitas
df_evaluacion.to_csv("Reporte_Punto_6_Evaluacion_P.csv", index=False)
print("-> Tabla completa guardada en 'Reporte_Punto_6_Evaluacion_P.csv'")

# Tabla de contingencia / resumen por clases como en la lámina 34
resumen_clases = []
total_patrones = 0
total_aciertos = 0
total_errores = 0

for c in sorted(df_P[clase_objetivo].unique()):
    sub_c = df_evaluacion[df_evaluacion['Clase real'] == c]
    n_patrones = len(sub_c)
    aciertos = (sub_c['Resultado'] == 'ACIERTO').sum()
    errores = (sub_c['Resultado'] == 'ERROR').sum()
    
    total_patrones += n_patrones
    total_aciertos += aciertos
    total_errores += errores
    
    resumen_clases.append({
        'Clase': str(c),
        'Patrones': n_patrones,
        'Aciertos': aciertos,
        'Errores': errores,
        'Efectividad': f"{(aciertos / n_patrones) * 100:.2f}%" if n_patrones > 0 else "0.00%"
    })

# Formato visual con cajas de ancho fijo
ancho_clase = 10
ancho_num = 16
ancho_pct = 15

linea_separadora = "+" + "-" * (ancho_clase + 2) + "+" + "-" * (ancho_num + 2) + "+" + "-" * (ancho_num + 2) + "+" + "-" * (ancho_num + 2) + "+" + "-" * (ancho_pct + 2) + "+"

print("\n" + linea_separadora)
print(f"| {'Clase':^{ancho_clase}} | {'Patrones en P':^{ancho_num}} | {'Aciertos':^{ancho_num}} | {'Errores':^{ancho_num}} | {'Efectividad':^{ancho_pct}} |")
print(linea_separadora)

for fila in resumen_clases:
    print(f"| {fila['Clase']:^{ancho_clase}} | {fila['Patrones']:^{ancho_num}} | {fila['Aciertos']:^{ancho_num}} | {fila['Errores']:^{ancho_num}} | {fila['Efectividad']:^{ancho_pct}} |")

print(linea_separadora)
pct_total = f"{(total_aciertos / total_patrones) * 100:.2f}%" if total_patrones > 0 else "0.00%"
print(f"| {'Total':^{ancho_clase}} | {total_patrones:^{ancho_num}} | {total_aciertos:^{ancho_num}} | {total_errores:^{ancho_num}} | {pct_total:^{ancho_pct}} |")
print(linea_separadora)

# =============================================================================
# PUNTO 7: MEDIDA DE DESEMPEÑO ACCURACY
# =============================================================================
total_patrones = len(df_evaluacion)
total_aciertos = (df_evaluacion['Resultado'] == 'ACIERTO').sum()
accuracy = total_aciertos / total_patrones

print("\n" + "=" * 85)
print("PUNTO 7: MEDIDA DE DESEMPEÑO (ACCURACY)")
print("=" * 85)
print(f"Aciertos totales: {total_aciertos} / {total_patrones}")
print(f"Accuracy:         {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("=" * 85)


nombre_archivo_reporte = "reporte_C45_resultados.txt"

with open(nombre_archivo_reporte, "w", encoding="utf-8") as f:
    # -------------------------------------------------------------------------
    # PUNTO 3: TABLA EN EL NODO RAÍZ
    # -------------------------------------------------------------------------
    f.write("=" * 90 + "\n")
    f.write(f"PUNTO 3: TABLA DE EVALUACIÓN EN NODO RAÍZ (Ganancia Promedio = {ganancia_promedio:.4f})\n")
    f.write("=" * 90 + "\n")
    f.write(df_p3[['Atributo', 'Tipo', 'Mejor t', 'Ganancia', '¿≥ promedio?', 'SplitInfo', 'GainRatio']].to_string(index=False))
    f.write(f"\n\n--> ATRIBUTO ELEGIDO COMO RAÍZ: {raiz_elegida['Atributo']}\n\n")

    # -------------------------------------------------------------------------
    # PUNTO 4: REDUCCIÓN DE IMPUREZA
    # -------------------------------------------------------------------------
    f.write("=" * 90 + "\n")
    f.write("PUNTO 4: REDUCCIÓN DE IMPUREZA EN NODO RAÍZ\n")
    f.write("=" * 90 + "\n")
    f.write(df_p4.to_string(index=False))
    f.write("\n\n--> ATRIBUTO ELEGIDO POR CADA MEDIDA:\n")
    f.write(f"* Entropía:            {df_p4.loc[df_p4['Δ Entropía'].idxmax(), 'Atributo']}\n")
    f.write(f"* Simpson-Gini:        {df_p4.loc[df_p4['Δ Gini'].idxmax(), 'Atributo']}\n")
    f.write(f"* Error Clasificación: {df_p4.loc[df_p4['Δ Error'].idxmax(), 'Atributo']}\n")
    f.write(f"* Mínima Probabilidad: {df_p4.loc[df_p4['Δ Mínimo'].idxmax(), 'Atributo']}\n\n")

# -------------------------------------------------------------------------
    # PUNTO 5: ESTRUCTURA COMPLETA DEL ÁRBOL
    # -------------------------------------------------------------------------
    f.write("=" * 90 + "\n")
    f.write("PUNTO 5: ESTRUCTURA Y REPRESENTACIÓN DEL ÁRBOL C4.5\n")
    f.write("=" * 90 + "\n")
    f.write("--- ESTRUCTURA COMPLETA DEL ÁRBOL ---\n")
    imprimir_arbol(arbol, sangria=0, archivo=f)
    f.write("\n\n")
    # -------------------------------------------------------------------------
    # PUNTO 6: EVALUACIÓN DE PATRONES DE PRUEBA Y TABLA RESUMEN
    # -------------------------------------------------------------------------
    f.write("=" * 90 + "\n")
    f.write("PUNTO 6: RESUMEN DE ACIERTOS Y ERRORES POR CLASE (CONJUNTO P)\n")
    f.write("=" * 90 + "\n")
    f.write(f"{'Clase':^10} | {'Patrones en P':^15} | {'Aciertos':^10} | {'Errores':^8}\n")
    f.write("-" * 52 + "\n")
    for fila in resumen_clases:
        f.write(f"{fila['Clase']:^10} | {fila['Patrones']:^15} | {fila['Aciertos']:^10} | {fila['Errores']:^8}\n")
    f.write("-" * 52 + "\n")
    f.write(f"{'Total':^10} | {total_patrones:^15} | {total_aciertos:^10} | {total_errores:^8}\n\n")

    # -------------------------------------------------------------------------
    # PUNTO 7: ACCURACY
    # -------------------------------------------------------------------------
    f.write("=" * 90 + "\n")
    f.write("PUNTO 7: MEDIDA DE DESEMPEÑO (ACCURACY)\n")
    f.write("=" * 90 + "\n")
    f.write(f"Aciertos totales: {total_aciertos} / {total_patrones}\n")
    f.write(f"Accuracy:         {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    f.write("=" * 90 + "\n\n")

    # Anexo opcional al final: el detalle patrón por patrón
    f.write("--- DETALLE DE CLASIFICACIÓN DE CADA PATRÓN DE PRUEBA (P) ---\n")
    f.write(df_evaluacion[['#', 'Clase real', 'Clase asignada', 'Resultado', 'Condiciones evaluadas (ruta)']].to_string(index=False))

print(f"\n[OK] ¡Reporte completo guardado exitosamente en '{nombre_archivo_reporte}'!")