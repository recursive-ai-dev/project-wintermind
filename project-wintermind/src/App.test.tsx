import { test, expect } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { App } from './App';

test('renders App and controls work', async () => {
  render(<App />);

  // start training
  const buttons = screen.getAllByRole('button');
  const startButton = buttons.find(b => b.textContent?.includes('START'));
  expect(startButton).toBeDefined();

  fireEvent.click(startButton!);
  await waitFor(() => {
    const stopButton = screen.getAllByRole('button').find(b => b.textContent?.includes('PAUSE'));
    expect(stopButton).toBeInTheDocument();
  });
});
