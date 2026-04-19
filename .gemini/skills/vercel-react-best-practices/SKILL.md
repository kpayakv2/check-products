# Vercel React Best Practices Skill

## Overview
Optimized guide for React and Next.js development within the `taxonomy-app` folder.

## Key Rules (by Priority)

### 1. Eliminating Waterfalls (CRITICAL) - Prefix: `async-`
- **Parallel Fetching:** Use `Promise.all()` for independent data fetches (e.g., fetching categories and product stats simultaneously).
- **Suspense Boundaries:** Use React Suspense to stream heavy dashboard components.

### 2. Bundle Size Optimization (CRITICAL) - Prefix: `bundle-`
- **Direct Imports:** Import UI components directly (e.g., `import { Button } from '@/components/ui/button'`) instead of through barrel files.
- **Dynamic Imports:** Use `next/dynamic` for heavy charts or visualizations in the taxonomy dashboard.

### 3. Re-render Optimization (MEDIUM) - Prefix: `rerender-`
- **Memoization:** Use `React.memo` and `useMemo` for expensive calculations like product similarity sorting.
- **Derived State:** Calculate derived state during render, not in `useEffect`.

## How to Apply
Invoke this skill when:
- Building or refactoring UI components in the `taxonomy-app`.
- Optimizing dashboard performance for large datasets.
- Reviewing frontend code for best practices.
