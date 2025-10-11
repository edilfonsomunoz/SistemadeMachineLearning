from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
from datetime import datetime
import base64

class ReportGenerator:
    def generate_pdf(self, report_data, session):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        story.append(Paragraph("Reporte de Análisis de Machine Learning", title_style))
        story.append(Spacer(1, 12))
        
        info_data = [
            ['Estudiante:', session.get('name', 'N/A')],
            ['Código:', session.get('student_code', 'N/A')],
            ['Email:', session.get('email', 'N/A')],
            ['Fecha:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        models = report_data.get('models', [])
        
        for idx, model in enumerate(models):
            if idx > 0:
                story.append(PageBreak())
            
            story.append(Paragraph(f"Modelo {idx + 1}: {model.get('model_type', 'N/A')}", heading_style))
            story.append(Spacer(1, 12))
            
            metrics = model.get('metrics', {})
            if metrics:
                story.append(Paragraph("Métricas del Modelo:", styles['Heading3']))
                metrics_data = [[k, str(v)] for k, v in metrics.items()]
                metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
                metrics_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
                ]))
                story.append(metrics_table)
                story.append(Spacer(1, 12))
            
            if 'plot' in model and model['plot']:
                story.append(Paragraph("Visualización:", styles['Heading3']))
                try:
                    img_data = base64.b64decode(model['plot'])
                    img_buffer = BytesIO(img_data)
                    img = Image(img_buffer, width=5*inch, height=2.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    story.append(Paragraph(f"Error al cargar imagen: {str(e)}", styles['Normal']))
            
            if 'importance_plot' in model and model['importance_plot']:
                story.append(Paragraph("Importancia de Variables:", styles['Heading3']))
                try:
                    img_data = base64.b64decode(model['importance_plot'])
                    img_buffer = BytesIO(img_data)
                    img = Image(img_buffer, width=5*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    story.append(Paragraph(f"Error al cargar imagen: {str(e)}", styles['Normal']))
            
            if 'feature_importance' in model:
                story.append(Paragraph("Importancia de Características:", styles['Heading3']))
                importance_data = [[k, f"{v:.4f}"] for k, v in model['feature_importance'].items()]
                if importance_data:
                    importance_table = Table(importance_data, colWidths=[3*inch, 2*inch])
                    importance_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                    ]))
                    story.append(importance_table)
        
        story.append(PageBreak())
        story.append(Paragraph("Conclusiones", heading_style))
        story.append(Paragraph(
            "Este reporte presenta el análisis de modelos de machine learning aplicados a los datos proporcionados. "
            "Los modelos incluyen técnicas de regresión, clasificación y árboles de decisión con sus respectivas métricas de evaluación.",
            styles['Normal']
        ))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
