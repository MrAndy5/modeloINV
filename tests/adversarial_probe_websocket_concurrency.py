"""
Adversarial WebSocket concurrency & thread safety stress test.
"""
import asyncio
import threading
import random
from unittest.mock import AsyncMock, MagicMock
from modeloinv.dashboard.websockets import ConnectionManager

def test_multithreaded_websocket_stress():
    manager = ConnectionManager()
    errors = []

    class MockWebSocket:
        def __init__(self, fail_rate=0.2):
            self.fail_rate = fail_rate
            self.accepted = False
            self.sent = []

        async def accept(self):
            self.accepted = True

        async def send_json(self, msg):
            if random.random() < self.fail_rate:
                raise RuntimeError("Simulated socket connection abort")
            self.sent.append(msg)

    def worker_thread(thread_id, num_cycles=50):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run_cycles():
            local_sockets = []
            for i in range(num_cycles):
                action = random.choice(["connect", "disconnect", "broadcast", "tick"])
                try:
                    if action == "connect" or not local_sockets:
                        ws = MockWebSocket(fail_rate=0.25)
                        await manager.connect(ws)
                        local_sockets.append(ws)
                    elif action == "disconnect":
                        ws = local_sockets.pop(random.randrange(len(local_sockets)))
                        manager.disconnect(ws)
                    elif action == "broadcast":
                        await manager.broadcast({"type": "stress", "thread": thread_id, "iter": i})
                    elif action == "tick":
                        await manager.broadcast_tick(
                            symbol="AAPL",
                            price=150.0 + random.random(),
                            portfolio_equity_eur=100.0 + random.random(),
                        )
                except Exception as e:
                    errors.append((thread_id, type(e), str(e)))

            # Clean up remaining
            for ws in local_sockets:
                try:
                    manager.disconnect(ws)
                except Exception as e:
                    errors.append((thread_id, type(e), str(e)))

        loop.run_until_complete(run_cycles())
        loop.close()

    threads = [threading.Thread(target=worker_thread, args=(tid,)) for tid in range(15)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Stress test finished. Errors encountered: {len(errors)}")
    if errors:
        for err in errors[:10]:
            print("Error sample:", err)
    assert len(errors) == 0, f"Encountered {len(errors)} concurrency errors during WebSocket stress!"
    print(f"Final active connections: {len(manager.active_connections)}")
    assert len(manager.active_connections) == 0, "Dangling connections after cleanup!"
    print("SUCCESS: ConnectionManager survived 750 concurrent multithreaded cycles with zero errors!")

if __name__ == "__main__":
    test_multithreaded_websocket_stress()
