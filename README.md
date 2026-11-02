# alfombra-fotos
Automatización interna de La Boutique de Sofii (FOTO-04): cuando llega una foto nueva de un diseño publicado,
GitHub Actions le pone el fondo de alfombra crema (principal) + azul, o el tapete navideño, y la entrega a Supabase.
Disparo: Apps Script "Sincronizar tienda" (repository_dispatch `fotos`), manual, o respaldo cada 6 h.
Secreto requerido: `SYNC_KEY`. El código no contiene llaves.
