import os
import stripe
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class StripeService:
    """Service for handling Stripe payments and checkout sessions."""
    
    def __init__(self):
        self.secret_key = os.environ.get('STRIPE_SECRET_KEY')
        if self.secret_key:
            stripe.api_key = self.secret_key
        else:
            logger.warning("STRIPE_SECRET_KEY not found in environment variables")
    
    def create_checkout_session(self, template_data: Dict[str, Any], success_url: str, cancel_url: str, customer_email: str = None) -> Dict[str, Any]:
        """Create a Stripe checkout session for template purchase."""
        try:
            if not self.secret_key:
                return {
                    'success': False,
                    'error': 'Stripe not configured. Please contact support.'
                }
            
            # Create checkout session
            session_data = {
                'payment_method_types': ['card'],
                'line_items': [{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': template_data['title'],
                            'description': template_data['description'],
                            'images': []  # Add template screenshots if available
                        },
                        'unit_amount': int(template_data['price'] * 100),  # Convert to cents
                    },
                    'quantity': 1,
                }],
                'mode': 'payment',
                'success_url': success_url,
                'cancel_url': cancel_url,
                'metadata': {
                    'template_slug': template_data['slug'],
                    'template_title': template_data['title']
                }
            }
            
            # Only add customer_email if provided
            if customer_email:
                session_data['customer_email'] = customer_email
                session_data['metadata']['customer_email'] = customer_email
                
            session = stripe.checkout.Session.create(**session_data)
            
            return {
                'success': True,
                'checkout_url': session.url,
                'session_id': session.id
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating checkout session: {e}")
            return {
                'success': False,
                'error': f'Payment setup failed: {str(e)}'
            }
        except Exception as e:
            logger.error(f"Error creating Stripe checkout session: {e}")
            return {
                'success': False,
                'error': f'Payment setup failed: {str(e)}'
            }
    
    def verify_payment(self, session_id: str) -> Dict[str, Any]:
        """Verify a payment session and get payment details."""
        try:
            if not self.secret_key:
                return {'success': False, 'error': 'Stripe not configured'}
            
            session = stripe.checkout.Session.retrieve(session_id)
            
            if session.payment_status == 'paid':
                return {
                    'success': True,
                    'paid': True,
                    'customer_email': session.customer_details.email if session.customer_details else None,
                    'amount_paid': session.amount_total / 100,  # Convert from cents
                    'template_slug': session.metadata.get('template_slug'),
                    'template_title': session.metadata.get('template_title')
                }
            else:
                return {
                    'success': True,
                    'paid': False,
                    'payment_status': session.payment_status
                }
                
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error verifying payment: {e}")
            return {'success': False, 'error': str(e)}
        except Exception as e:
            logger.error(f"Error verifying Stripe payment: {e}")
            return {'success': False, 'error': str(e)}
    
    def handle_webhook(self, payload: str, sig_header: str) -> Dict[str, Any]:
        """Handle Stripe webhook events."""
        try:
            webhook_secret = os.environ.get('STRIPE_WEBHOOK_SECRET')
            if not webhook_secret:
                logger.warning("STRIPE_WEBHOOK_SECRET not configured")
                return {'success': False, 'error': 'Webhook not configured'}
            
            event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
            
            if event['type'] == 'checkout.session.completed':
                session = event['data']['object']
                return {
                    'success': True,
                    'event_type': 'payment_completed',
                    'session_id': session['id'],
                    'customer_email': session.get('customer_details', {}).get('email'),
                    'amount_paid': session['amount_total'] / 100,
                    'template_slug': session.get('metadata', {}).get('template_slug'),
                    'template_title': session.get('metadata', {}).get('template_title')
                }
            
            return {'success': True, 'event_type': event['type']}
            
        except Exception as e:
            logger.error(f"Error handling Stripe webhook: {e}")
            return {'success': False, 'error': str(e)}
    
    def is_configured(self) -> bool:
        """Check if Stripe is properly configured."""
        return bool(self.secret_key)

# Singleton instance
stripe_service = StripeService()