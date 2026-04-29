# Health Monitoring Dashboard

A React + Vite health monitoring dashboard with 3D watch visualization, real-time vitals simulation, and charting.

## Tech Stack

- **React 18** + **TypeScript**
- **Vite 6** (standalone, no pnpm workspace)
- **Tailwind CSS v4** via `@tailwindcss/vite`
- **PostCSS** + **autoprefixer** (no lightningcss)
- **shadcn/ui** components (Radix UI)
- **Three.js** + **@react-three/fiber** for 3D watch
- **Recharts** for analytics charts
- **Framer Motion** for animations
- **Wouter** for client-side routing
- **TanStack Query** for data management

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## Available Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start dev server on port 5173 |
| `npm run build` | Type-check and build for production |
| `npm run preview` | Preview production build locally |
| `npm run typecheck` | Run TypeScript type checking only |

## Project Structure

```
src/
├── components/
│   ├── ui/          # shadcn/ui components
│   ├── AnalyticsPanel.tsx
│   ├── Layout.tsx
│   ├── ThemeToggle.tsx
│   ├── VitalsInput.tsx
│   └── Watch3D.tsx
├── context/
│   ├── ThemeContext.tsx
│   └── VitalsContext.tsx
├── hooks/
│   ├── use-mobile.tsx
│   └── use-toast.ts
├── lib/
│   ├── utils.ts
│   └── vitalsLogic.ts
├── pages/
│   ├── About.tsx
│   ├── DashboardLive.tsx
│   ├── DetailedAnalytics.tsx
│   ├── History.tsx
│   └── not-found.tsx
├── App.tsx
├── main.tsx
└── index.css
```

## Notes

- All vitals data is **simulated locally** via React Context — no backend required.
- Dark/light theme is supported with a toggle in the header.
- The 3D watch component uses Three.js and requires a fixed-size container to render correctly.
