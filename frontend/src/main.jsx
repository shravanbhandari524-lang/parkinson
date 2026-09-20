import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import '@fontsource-variable/inter';
import './index.css';
import App from './App.jsx';
import { AssessmentProvider } from './hooks/useAssessment.jsx';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <AssessmentProvider>
        <App />
      </AssessmentProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
