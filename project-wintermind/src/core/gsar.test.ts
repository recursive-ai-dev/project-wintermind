import { test, expect } from 'vitest';
import { GSARModule } from './gsar';

test('GSARModule reason', () => {
  const gsar = new GSARModule();
  const tokens = ['winter', 'wind', 'blows', 'cold'];
  const result = gsar.reason(tokens);

  expect(result.segments.length).toBeGreaterThan(0);
});
