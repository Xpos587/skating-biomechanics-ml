import { setupServer } from "msw/node"
import { afterAll, afterEach } from "vitest"
import { handlers } from "./handlers"

// MSW server for Vitest
export const server = setupServer(...handlers)

// Setup and teardown for Vitest
export function setupMSW() {
  // Establish API mocking before all tests
  server.listen({ onUnhandledRequest: "error" })

  // Reset any request handlers that we may add during the tests
  afterEach(() => {
    server.resetHandlers()
  })

  // Clean up after the tests are finished
  afterAll(() => {
    server.close()
  })
}
