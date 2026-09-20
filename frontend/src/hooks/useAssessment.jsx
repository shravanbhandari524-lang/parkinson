import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import { analyzeFiles as analyzeFilesApi } from '../services/api.js';
import { ApiError } from '../services/api.js';

/**
 * Global assessment state shared between the Assessment and Results pages:
 * status, the normalized API prediction, upload progress, and errors.
 */
const AssessmentContext = createContext(null);

export function AssessmentProvider({ children }) {
  const [status, setStatus] = useState('idle'); // idle | analyzing | success | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [analyzedFileNames, setAnalyzedFileNames] = useState({});
  const abortRef = useRef(null);

  const analyze = useCallback(async (files) => {
    setStatus('analyzing');
    setError(null);
    setResult(null);
    setProgress(0);
    setAnalyzedFileNames(
      Object.fromEntries(
        Object.entries(files ?? {})
          .filter(([, file]) => Boolean(file))
          .map(([key, file]) => [key, file.name]),
      ),
    );

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const normalized = await analyzeFilesApi(files, {
        signal: controller.signal,
        onUploadProgress: (event) => {
          if (!event.total) return;
          setProgress(Math.min(99, Math.round((event.loaded / event.total) * 100)));
        },
      });
      setProgress(100);
      setResult(normalized);
      setStatus('success');
      return normalized;
    } catch (caught) {
      const apiError =
        caught instanceof ApiError
          ? caught
          : new ApiError(caught?.message ?? 'Unexpected error during analysis.');
      if (controller.signal.aborted || apiError.code === 'ERR_CANCELED') {
        setStatus('idle');
        return null;
      }
      setError(apiError);
      setStatus('error');
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus('idle');
    setResult(null);
    setError(null);
    setProgress(0);
    setAnalyzedFileNames({});
  }, []);

  const value = useMemo(
    () => ({ status, result, error, progress, analyzedFileNames, analyze, reset }),
    [status, result, error, progress, analyzedFileNames, analyze, reset],
  );

  return <AssessmentContext.Provider value={value}>{children}</AssessmentContext.Provider>;
}

export function useAssessment() {
  const context = useContext(AssessmentContext);
  if (!context) {
    throw new Error('useAssessment must be used within an AssessmentProvider');
  }
  return context;
}

export { AssessmentContext };
