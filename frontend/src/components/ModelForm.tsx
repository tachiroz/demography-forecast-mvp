import { useState } from 'react';

interface Props {
  selected: string;
  onSelect: (m: string) => void;
  onTrained: () => void;
  onForecast: (data: { Year: number[]; y_pred: number[] }) => void;
}

export default function ModelForm({
  selected,
  onSelect,
  onTrained,
  onForecast,
}: Props) {
  const [loading, setLoading] = useState(false);
  const [years, setYears] = useState(5); // 1‒22 лет вперёд

  const train = async () => {
    setLoading(true);
    await fetch('/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: selected, params: {} }),
    });
    setLoading(false);
    onTrained();
  };

  const forecast = async () => {
    const r = await fetch(`/forecast/${selected}?years=${years}`);
    const data = await r.json();
    if (data.status === 'error') {
      alert(data.detail);
    } else {
      onForecast(data); // Year[], y_pred[]
    }
  };

  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ marginRight: 8 }}>Model:</label>

      <select value={selected} onChange={e => onSelect(e.target.value)}>
        <option value="sarimax">SARIMAX (Births)</option>
        <option value="sarimax_pop">SARIMAX-POP (Population)</option>
        <option value="prophet">Prophet</option>
        <option value="xgb">XGBoost</option>
        <option value="cat">CatBoost</option>
      </select>

      <button
        style={{ marginLeft: 12 }}
        onClick={train}
        disabled={loading}
      >
        {loading ? 'Training…' : 'Train'}
      </button>

      {/* слайдер количества лет */}
      <span style={{ marginLeft: 24 }}>Forecast:</span>
      <input
        type="range"
        min={1}
        max={22}
        value={years}
        onChange={e => setYears(+e.target.value)}
        style={{ verticalAlign: 'middle', width: 120, margin: '0 8px' }}
      />
      <span>{years}</span>

      <button style={{ marginLeft: 8 }} onClick={forecast}>
        Predict
      </button>
    </div>
  );
}
