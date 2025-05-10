import { useState } from 'react';
import ModelForm from './components/ModelForm';
import MetricsTable from './components/MetricsTable';
import ForecastChart from './components/ForecastChart';

export default function App() {
  const [reload, setReload] = useState(0);
  const handleTrained = () => setReload(r => r + 1);

  return (
    <div style={{ padding: 20 }}>
      <h1>Demography Forecast MVP</h1>
      <ModelForm onTrained={handleTrained} />
      <MetricsTable key={reload} model="sarimax" />
      <ForecastChart key={reload} model="sarimax" />
    </div>
  );
}
