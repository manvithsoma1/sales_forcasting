# Fix: "Descriptors cannot be created directly" on Streamlit Cloud

If you see this protobuf error after deploying, add an environment variable in Streamlit Cloud:

## Steps

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and open your app
2. Click the **⋮** (three dots) → **Settings**
3. Open the **"Secrets"** tab (or **"Advanced"** / **"Environment variables"**)
4. Add this secret (Streamlit exposes root secrets as env vars):

   ```toml
   PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
   ```

   Or if it asks for key/value:
   - **Key:** `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION`
   - **Value:** `python`

5. Click **Save**
6. Click **Reboot app** (or **Restart**) in the app menu

This uses pure-Python protobuf parsing, which avoids the error. The app may be slightly slower but will run correctly.
