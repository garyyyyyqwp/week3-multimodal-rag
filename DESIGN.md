# Vercel Design System — for Multimodal RAG Platform

## 1. Visual Theme & Atmosphere
- **Dark-first** data-dense developer platform. Fiercely minimal, precise geometry.
- All surfaces slightly lifted with multi-layer shadows rather than heavy borders.
- High contrast text on deep black backgrounds. Accent colors used sparingly for
  interactive elements and status indicators.
- Console-inspired code blocks and technical labels.

## 2. Color System

| Token | HEX | Usage |
|---|---|---|
| `--bg-primary` | `#000000` | Page background |
| `--bg-secondary` | `#0a0a0a` | Card surfaces, sidebar |
| `--bg-tertiary` | `#141414` | Input fields, hover states |
| `--bg-elevated` | `#1a1a1a` | Modals, dropdowns, tooltips |
| `--text-primary` | `#ededed` | Headings, body text |
| `--text-secondary` | `#888888` | Labels, meta text |
| `--text-tertiary` | `#666666` | Placeholders, disabled |
| `--accent-blue` | `#0a72ef` | Primary buttons, links, focus |
| `--accent-red` | `#ff5b4f` | Errors, delete, critical |
| `--accent-green` | `#00d684` | Success, indexed, completed |
| `--accent-amber` | `#f5a623` | Warnings, running status |
| `--accent-purple` | `#7928ca` | Experiment badges, code |
| `--accent-pink` | `#eb367f` | Preview labels |
| `--border-subtle` | `rgba(255,255,255,0.08)` | Card borders, dividers |
| `--border-default` | `rgba(255,255,255,0.12)` | Input borders |
| `--border-strong` | `rgba(255,255,255,0.20)` | Focus rings |

## 3. Typography Hierarchy

| Level | Size | Weight | Letter Spacing | Usage |
|---|---|---|---|---|
| Hero | 48px | 600 | -2.4px | Page titles |
| H1 | 32px | 600 | -1.6px | Section headers |
| H2 | 20px | 600 | -0.6px | Card titles |
| Body | 16px | 400 | -0.16px | Main text, descriptions |
| Body-Small | 14px | 400 | -0.08px | Secondary text, table cells |
| Label | 12px | 500 | +0.24px | Badges, tags, metadata |
| Mono | 13px | 400 | 0px | Code, token counts, IDs |
| Micro | 10px | 500 | +0.48px | Status dots, micro badges |

Font family: `"Geist", "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`
Mono: `"Geist Mono", "JetBrains Mono", "Fira Code", monospace`
All text: `font-feature-settings: "liga" 1;`

## 4. Component Styles

### Buttons
- **Primary**: bg `--accent-blue`, text white, 14px/500, py-8 px-16, radius 6px
- **Secondary**: bg transparent, border `--border-default`, text `--text-primary`, radius 6px
- **Danger**: bg transparent, border `--accent-red`, text `--accent-red`
- **Ghost**: bg transparent, text `--text-secondary`, hover bg `--bg-tertiary`
- All buttons: `transition: all 0.15s ease`, focus ring: `0 0 0 2px --accent-blue`

### Cards
- bg `--bg-secondary`, border: none
- `box-shadow: 0px 0px 0px 1px rgba(255,255,255,0.06), 0px 2px 8px rgba(0,0,0,0.4)`
- radius 8px, padding 24px
- Hover: `box-shadow: 0px 0px 0px 1px rgba(255,255,255,0.10), 0px 4px 16px rgba(0,0,0,0.5)`

### Inputs
- bg `--bg-tertiary`, border `--border-default`, text `--text-primary`, placeholder `--text-tertiary`
- radius 6px, height 40px, px-12
- Focus: border `--accent-blue`, `box-shadow: 0 0 0 2px rgba(10,114,239,0.25)`

### Tables
- Header: text `--text-secondary`, 12px/500, uppercase tracking +0.48px
- Cells: text `--text-primary`, 14px
- Row divider: `border-bottom: 1px solid --border-subtle`
- Hover: bg `--bg-tertiary`
- Numeric cells: right-aligned, mono font

### Status Badges
- Pill shape, 10px/500, all-caps
- Active: bg `rgba(0,214,132,0.12)`, text `--accent-green`
- Running: bg `rgba(245,166,35,0.12)`, text `--accent-amber`
- Error: bg `rgba(255,91,79,0.12)`, text `--accent-red`

## 5. Layout Principles
- Max content width 1200px, centered
- Sidebar: 240px fixed, main content flex-1
- Page padding: 48px
- Card grid: 2-3 columns, 16px gap
- Vertical rhythm: multiples of 8px (8, 16, 24, 32, 48, 64)

## 6. Shadows & Elevation
```
Level 0: none (flat)
Level 1: 0px 0px 0px 1px rgba(255,255,255,0.06), 0px 2px 8px rgba(0,0,0,0.4)
Level 2: 0px 0px 0px 1px rgba(255,255,255,0.08), 0px 4px 16px rgba(0,0,0,0.5)
Level 3: 0px 0px 0px 1px rgba(255,255,255,0.10), 0px 8px 32px rgba(0,0,0,0.6)
```

## 7. Do's and Don'ts
- ✅ Dark backgrounds always — never pure white cards
- ✅ Multi-layer shadows for depth, not borders
- ✅ Mono font for all numbers, tokens, code, IDs
- ✅ Negative letter-spacing on headings
- ✅ Accent colors only on interactive/highlight elements
- ❌ No pure `#fff` text — use `#ededed`
- ❌ No gradient backgrounds on cards
- ❌ No rounded corners larger than 8px (except pill badges)
- ❌ No drop shadows without the `0px 0px 0px 1px` border ring

## 8. Responsive Behavior
- Below 768px: sidebar collapses to top nav
- Below 480px: cards go single column, padding reduces to 16px
- Tables: horizontal scroll on mobile

## 9. AI Agent Guidelines
- Generate semantic HTML5 with CSS custom properties
- All interactive elements must have visible focus states
- Use `<table>` for data tables, not div grids
- Status indicators: always accompanied by text labels
- Loading states: subtle pulse animation, never spinning borders
- Empty states: centered icon + description + CTA
- Error states: red border-left accent on the affected card
