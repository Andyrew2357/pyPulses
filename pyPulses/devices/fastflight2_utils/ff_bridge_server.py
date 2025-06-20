import json
import sys
import traceback
from fastflight32 import FastFlight32
from typing import Any, Dict

class FastFlightBridge():
    def __init__(self):
        self.ff2 = None

    def handle_request(self, request: Dict[str, Any]):
        """Handle a single request and return response"""

        try:
            method = request['method']
            args = request.get('args', [])
            kwargs = request.get('kwargs', {})

            # Initialize FastFlight instance if needed
            if self.ff2 is None and method != 'init':
                self.ff2 = FastFlight32()

            # Handle method calls
            if method == 'init':
                self.ff2 = FastFlight32()
                return {'success': True, 'result': None}
                
            elif method == 'del':
                if self.ff2:
                    del self.ff2
                    self.ff2 = None
                return {'success': True, 'result': None}
            
            elif hasattr(self.ff2, method):
                result = getattr(self.ff2, method)(*args, **kwargs)
                return {'success': True, 'result': result}
            
            else:
                return {'success': False, 
                        'error'  : f'Unknown method: {method}'
                    }
            
        except Exception as e:
            return {
                'success'   : False,
                'error'     : str(e),
                'traceback' : traceback.format_exc()
            }
        
    def run(self):
        """
        Main message loop. Read JSON requests from stdin, write to stdout.
        """

        try:
            while True:
                line = sys.stdin.readline().strip()
                
                if not line or line == 'QUIT':
                    break

                try:
                    request = json.loads(line)
                    response = self.handle_request(request)
                    print(json.dumps(response), flush = True)

                except json.JSONDecodeError as e:
                    error_response = {
                        'success': False, 
                        'error'  : f'JSON decode error: {str(e)}'
                    }
                    print(json.dumps(error_response), flush = True)

        except KeyboardInterrupt:
            pass
        
        finally:
            if self.ff2:
                try:
                    if self.ff2.is_connected():
                        self.ff2.close()
                except:
                    pass

if __name__ == '__main__':
    bridge = FastFlightBridge()
    bridge.run()
