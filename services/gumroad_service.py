import os
import requests
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class GumroadService:
    """Service for integrating with Gumroad API for template sales and revenue tracking."""
    
    def __init__(self):
        self.api_key = os.environ.get('GUMROAD_API_KEY')
        self.seller_id = os.environ.get('GUMROAD_SELLER_ID', 'cesarano1')
        self.base_url = 'https://api.gumroad.com/v2'
        
        if not self.api_key:
            logger.error("GUMROAD_API_KEY not found in environment variables")
            raise ValueError("Gumroad API key is required")
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request to Gumroad API."""
        url = f"{self.base_url}/{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=headers, json=data)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Gumroad API request failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_product(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new product on Gumroad for a template."""
        product_data = {
            'name': template_data['title'],
            'price': int(template_data['price'] * 100),  # Convert to cents
            'description': template_data['description'],
            'summary': template_data.get('long_description', template_data['description'])[:160],
            'tags': ','.join(template_data.get('tags', [])),
            'file_url': template_data.get('download_url', ''),
            'variants': [],
            'recurrences': None,  # One-time purchase
            'require_shipping': False,
            'customizable_price': None,
            'shown_on_profile': True
        }
        
        try:
            response = self._make_request('POST', 'products', data=product_data)
            if response.get('success'):
                logger.info(f"Successfully created Gumroad product: {template_data['title']}")
                return {
                    'success': True,
                    'product_id': response['product']['id'],
                    'product_url': response['product']['short_url'],
                    'gumroad_data': response['product']
                }
            else:
                logger.error(f"Failed to create Gumroad product: {response}")
                return {'success': False, 'error': response.get('error', 'Unknown error')}
                
        except Exception as e:
            logger.error(f"Error creating Gumroad product: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_product(self, product_id: str, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing Gumroad product."""
        product_data = {
            'name': template_data['title'],
            'price': int(template_data['price'] * 100),
            'description': template_data['description'],
            'summary': template_data.get('long_description', template_data['description'])[:160],
            'tags': ','.join(template_data.get('tags', []))
        }
        
        try:
            response = self._make_request('PUT', f'products/{product_id}', data=product_data)
            if response.get('success'):
                logger.info(f"Successfully updated Gumroad product: {product_id}")
                return {'success': True, 'gumroad_data': response['product']}
            else:
                logger.error(f"Failed to update Gumroad product: {response}")
                return {'success': False, 'error': response.get('error', 'Unknown error')}
                
        except Exception as e:
            logger.error(f"Error updating Gumroad product: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_sales_data(self, after_date: Optional[str] = None, before_date: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve sales data from Gumroad."""
        params = {}
        if after_date:
            params['after'] = after_date
        if before_date:
            params['before'] = before_date
        
        try:
            response = self._make_request('GET', 'sales', params=params)
            if response.get('success'):
                return {
                    'success': True,
                    'sales': response.get('sales', []),
                    'total_revenue': sum(float(sale.get('price', 0)) for sale in response.get('sales', [])),
                    'total_sales': len(response.get('sales', []))
                }
            else:
                logger.error(f"Failed to retrieve Gumroad sales data: {response}")
                return {'success': False, 'error': response.get('error', 'Unknown error')}
                
        except Exception as e:
            logger.error(f"Error retrieving Gumroad sales data: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_product_sales(self, product_id: str) -> Dict[str, Any]:
        """Get sales data for a specific product."""
        try:
            params = {'product_id': product_id}
            response = self._make_request('GET', 'sales', params=params)
            
            if response.get('success'):
                sales = response.get('sales', [])
                return {
                    'success': True,
                    'sales_count': len(sales),
                    'total_revenue': sum(float(sale.get('price', 0)) for sale in sales),
                    'recent_sales': sales[:5]  # Last 5 sales
                }
            else:
                return {'success': False, 'error': response.get('error', 'Unknown error')}
                
        except Exception as e:
            logger.error(f"Error retrieving product sales: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_purchase_link(self, template_slug: str, template_price: float) -> str:
        """Generate a direct purchase link for a template."""
        # For now, generate a direct Gumroad link
        # This can be enhanced with product-specific links once products are created
        base_link = f"https://{self.seller_id}.gumroad.com/l/{template_slug}"
        return base_link
    
    def create_discount_code(self, product_id: str, discount_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a discount code for a product."""
        try:
            response = self._make_request('POST', f'products/{product_id}/discount_codes', data=discount_data)
            if response.get('success'):
                return {'success': True, 'discount_code': response['discount_code']}
            else:
                return {'success': False, 'error': response.get('error', 'Unknown error')}
                
        except Exception as e:
            logger.error(f"Error creating discount code: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_revenue_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive revenue analytics."""
        try:
            # Calculate date range
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get sales data
            sales_data = self.get_sales_data(
                after_date=start_date.strftime('%Y-%m-%d'),
                before_date=end_date.strftime('%Y-%m-%d')
            )
            
            if not sales_data.get('success'):
                return sales_data
            
            sales = sales_data.get('sales', [])
            
            # Calculate analytics
            total_revenue = sum(float(sale.get('price', 0)) for sale in sales)
            total_sales = len(sales)
            avg_sale_value = total_revenue / total_sales if total_sales > 0 else 0
            
            # Group by product
            product_performance = {}
            for sale in sales:
                product_id = sale.get('product_id', 'unknown')
                product_name = sale.get('product_name', 'Unknown Product')
                
                if product_id not in product_performance:
                    product_performance[product_id] = {
                        'name': product_name,
                        'sales': 0,
                        'revenue': 0.0
                    }
                
                product_performance[product_id]['sales'] += 1
                product_performance[product_id]['revenue'] += float(sale.get('price', 0))
            
            return {
                'success': True,
                'period_days': days,
                'total_revenue': round(total_revenue, 2),
                'total_sales': total_sales,
                'average_sale_value': round(avg_sale_value, 2),
                'product_performance': product_performance,
                'growth_metrics': {
                    'revenue_target': 15000,  # $15K target
                    'progress_percentage': min(100, (total_revenue / 15000) * 100),
                    'projected_monthly': (total_revenue / days) * 30 if days > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating revenue analytics: {e}")
            return {'success': False, 'error': str(e)}

# Singleton instance
gumroad_service = GumroadService()