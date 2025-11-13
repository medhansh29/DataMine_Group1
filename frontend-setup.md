# Frontend Setup Guide

## Architecture Overview

```
┌─────────────────┐         HTTP Requests         ┌──────────────────┐
│  React Frontend │  ──────────────────────────>  │  FastAPI Backend │
│   (Vercel)      │                                │  (Railway/Render)│
└─────────────────┘                                └──────────────────┘
```

**Important:** 
- **Frontend (React)**: Deploy on Vercel ✅
- **Backend (FastAPI)**: Deploy on Railway, Render, or Fly.io (NOT Vercel - Vercel doesn't support long-running Python apps)

## Option 1: React in Same Repo (Recommended for small projects)

### Structure:
```
DataMine_Group1/
├── app.py                 # FastAPI backend
├── frontend/              # React app
│   ├── src/
│   ├── package.json
│   └── ...
└── ...
```

### Steps:
1. Create React app in `frontend/` folder
2. Deploy backend separately (Railway/Render)
3. Deploy frontend to Vercel (point to `frontend/` folder)

## Option 2: Separate Repos (Recommended for larger projects)

- **Backend repo**: This current repo
- **Frontend repo**: Separate React repo
- Deploy independently

## Quick Start: Create React Frontend

### 1. Create React App (in same repo)

```bash
# In your project root
npx create-react-app frontend
# OR use Vite (faster)
npm create vite@latest frontend -- --template react
```

### 2. Install dependencies

```bash
cd frontend
npm install axios  # For API calls
```

### 3. Create API service

Create `frontend/src/api.js`:

```javascript
import axios from 'axios';

// Change this to your deployed backend URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getCSVData = async () => {
  const response = await api.get('/csv');
  return response.data;
};

export default api;
```

### 4. Create component to display data

Create `frontend/src/components/DataTable.jsx`:

```javascript
import { useState, useEffect } from 'react';
import { getCSVData } from '../api';

function DataTable() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const result = await getCSVData();
        setData(result);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <h1>ZTF Objects Data</h1>
      <table>
        <thead>
          <tr>
            <th>OID</th>
            <th>Peak Luminosity</th>
            <th>R²</th>
            <th>ML Score</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr key={idx}>
              <td>{row.oid}</td>
              <td>{row.peak_luminosity}</td>
              <td>{row.r_squared}</td>
              <td>{row.ML_score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default DataTable;
```

### 5. Update App.js

```javascript
import DataTable from './components/DataTable';

function App() {
  return (
    <div className="App">
      <DataTable />
    </div>
  );
}

export default App;
```

### 6. Environment variables

Create `frontend/.env`:
```
REACT_APP_API_URL=http://localhost:8000
```

For production (Vercel), set environment variable:
- `REACT_APP_API_URL=https://your-backend-url.railway.app`

## Deployment

### Backend (FastAPI) - Railway/Render

1. **Railway** (easiest):
   - Sign up at railway.app
   - New Project → Deploy from GitHub
   - Select this repo
   - Railway auto-detects FastAPI
   - Add environment variables if needed
   - Get your backend URL (e.g., `https://your-app.railway.app`)

2. **Render** (free tier available):
   - Sign up at render.com
   - New Web Service
   - Connect GitHub repo
   - Build command: `pip install -r requirements-py310-windows.txt && uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Frontend (React) - Vercel

1. Push code to GitHub
2. Go to vercel.com
3. Import project
4. Set root directory to `frontend/` (if React is in subfolder)
5. Add environment variable: `REACT_APP_API_URL=https://your-backend-url.railway.app`
6. Deploy!

## Testing Locally

### Terminal 1: Backend
```bash
uvicorn app:app --reload --port 8000
```

### Terminal 2: Frontend
```bash
cd frontend
npm start
```

Visit: http://localhost:3000

## Update CORS in app.py

After deploying backend, update CORS origins:

```python
allow_origins=[
    "http://localhost:3000",
    "https://your-frontend.vercel.app",  # Add your Vercel URL
]
```

