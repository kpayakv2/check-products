require('@testing-library/jest-dom')

// Note: database setup จะทำใน individual test files
// เพราะ setup file นี้ไม่รองรับ async imports

// Increase timeout for database operations
jest.setTimeout(30000)

// Mock console methods to reduce noise in tests (optional)
// global.console = {
//   ...console,
//   warn: jest.fn(),
//   error: jest.fn(),
// }
