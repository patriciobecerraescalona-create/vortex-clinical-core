\# BITÁCORA DEL PROYECTO



\## 2026-01-28 21:30 – 23:55 CLT

Participantes: Patricio, Pepe, Claude Code



Trabajo realizado:

\- Se validó el flujo de trabajo humano + Claude Code.

\- Se corrigió bug de endpoint en LAB (/api/procedures/voice → /lab).

\- Se agregó panel visual de estado KAI en el frontend LAB.



Hallazgos clave:

\- El backend ya entregaba toda la información necesaria.

\- El problema era de observabilidad, no de lógica.



Pendiente:

\- Diseñar modo simulación del LAB cuando el backend no esté disponible.



Cierre de sesión:

"Ok, dejamos esto hasta mañana."


## 2026-01-28 17:00 – 17:30 CLT

Participantes: Usuario, Claude Code

Trabajo realizado:
- Implementado evento startup en main.py para crear tablas automáticamente (Base.metadata.create_all).
- Agregado campo user_id a LabPayload con validación explícita (error LAB_MISSING_USER_ID).
- Modificado procedures.py para pasar user_id a VoiceEvent y MemoryNode.
- Actualizado frontend LAB con input visible para user_id y selector de rol.

Decisiones de diseño:
- En producción, user_id vendrá desde SGMI.
- En entorno LAB, user_id debe ser provisto explícitamente por el frontend.
- El backend NO inventa ni simula identidades.

Estado actual:
- Backend estable (Docker + PostgreSQL + FastAPI).
- Tablas se crean automáticamente al iniciar.
- Validación de user_id funciona correctamente.

Pendiente:
- Error LAB_MISSING_USER_ID persiste por caché del navegador (HTML antiguo).
- Solución: forzar recarga sin caché (Ctrl+Shift+R) o abrir en incógnito.

Próximos pasos:
1. Verificar que el frontend cargue correctamente tras limpiar caché.
2. Probar flujo completo LAB → DB con user_id válido.
3. Considerar modo simulación LAB (pendiente de sesión anterior).

Cierre de sesión:
Sesión cerrada intencionalmente con punto pendiente de resolución (caché frontend).

